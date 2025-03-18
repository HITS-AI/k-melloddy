import pandas as pd
import numpy as np
import os
import multiprocessing
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import re

from argparse import ArgumentParser
from typing import Iterable, List, Tuple, Dict, Any, Optional
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from scipy.stats import normaltest
from functools import partial

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The logger will be configured with file handler in main

def recognize_task_type(each_df, activity_col='Measurement_Value'):
    """
    Analyze the activity value column of a dataframe to determine the task type (classification or regression).
    
    Parameters:
    -----------
    each_df : pandas.DataFrame
        Dataframe to analyze
    activity_col : str
        Name of the activity value column
    
    Returns:
    --------
    str : 'classification' or 'regression'
    """
    # Check if column exists
    if activity_col not in each_df.columns:
        logger.warning(f"Activity column '{activity_col}' not found. Defaulting to classification.")
        return "classification"
    
    # If 'Not specified' or string values exist, it's a classification task
    if 'Not specified' in each_df[activity_col].values or \
       each_df[activity_col].astype(str).str.contains('^[a-zA-Z]').any():
        return "classification"
    
    # Check if all values can be converted to numbers
    try:
        # Try to convert to numeric, excluding comparison operators (>, <)
        numeric_values = pd.to_numeric(each_df[activity_col].astype(str).str.replace(r'^[<>]\s*', '', regex=True), errors='coerce')
        if numeric_values.isna().sum() > 0:
            # If values that can't be converted to numbers exist, it's a classification task
            return "classification"
        # Check if only 0 or 1 values exist
        unique_values = numeric_values.dropna().unique()
        if set(unique_values) <= {0, 1}:
            return "classification"
    except:
        return "classification"
    
    # Default to regression task
    return "regression"


class TrainingQuorumError(Exception):
    def __init__(self, task_type, message=None):
        self.task_type = task_type
        if message is None:
            if task_type.lower() == 'classification':
                message = "Training quorum is 25 actives and 25 inactives per task."
            elif task_type.lower() == 'regression':
                message = "Training quorum is 50 data points out of which 25 uncensored per task."
            else:
                message = "Training quorum requirement not met. At least 50 data points needed"
        super().__init__(message)


class DataInspector:
    def __init__(self,
                 input_path:str,
                 smiles_column:str='SMILES_Structure_Parent',
                 activity_column:str='Measurement_Value',
                 condition_columns:list=['Test', 'Test_Type', 'Test_Subject', 'Measurement_Type']
                 ):
        self.smiles_col = smiles_column
        self.activity_col = activity_column
        self.condition_columns = condition_columns
        self.df = self.load_data(input_path)
    
    def load_data(self, input_path):
        # Check if file exists
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        # Determine file type based on extension
        file_extension = os.path.splitext(input_path)[1].lower()
        
        if file_extension == '.csv':
            logger.info(f"Loading CSV file: {input_path}")
            df = pd.read_csv(input_path).fillna("Not specified")
        elif file_extension in ['.xlsx', '.xls']:
            logger.info(f"Loading Excel file: {input_path}")
            # Read both ADMET and PK sheets if they exist
            excel_file = pd.ExcelFile(input_path)
            sheets = excel_file.sheet_names
            
            dfs = []
            if 'ADMET' in sheets:
                logger.info("Reading ADMET sheet")
                admet_df = pd.read_excel(excel_file, sheet_name='ADMET').fillna("Not specified")
                dfs.append(admet_df)
            
            if 'PK' in sheets:
                logger.info("Reading PK sheet")
                pk_df = pd.read_excel(excel_file, sheet_name='PK').fillna("Not specified")
                dfs.append(pk_df)
                
            if not dfs:
                # If no specific sheets found, read the first sheet
                logger.warning(f"No ADMET or PK sheets found in {input_path}. Reading first sheet.")
                df = pd.read_excel(excel_file, sheet_name=0).fillna("Not specified")
            else:
                # Combine all sheets
                df = pd.concat(dfs, ignore_index=True)
                logger.info(f"Combined data from sheets. Total rows: {len(df)}")
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Please use CSV or Excel files.")
        
        # Validate required columns
        if self.smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{self.smiles_col}' not found in data.")
        
        # Process activity column if it exists
        if self.activity_col in df.columns:
            try:
                df[self.activity_col] = df[self.activity_col].astype(str)
                
                numeric_pattern = r'^[<>]?\s*\d+\.?\d*$'
                non_numeric_mask = ~df[self.activity_col].str.match(numeric_pattern)
                
                if non_numeric_mask.any():
                    logger.info(f"Activity column '{self.activity_col}' contains non-numeric values. These will be handled as categorical.")
            except Exception as e:
                logger.warning(f"Error analyzing activity column: {e}")
        else:
            logger.warning(f"Activity column '{self.activity_col}' not found in data. Some functionality may be limited.")

        # Replace special characters in column values
        for col in df.columns:
            if df[col].dtype == 'object':  # Only process string columns
                # Replace μ (micro) with u
                df[col] = df[col].astype(str).str.replace('μ', 'u')
                # Replace other potentially problematic unicode characters
                df[col] = df[col].apply(lambda x: self._replace_special_chars(x) if isinstance(x, str) else x)
                
        # Ensure all condition columns exist
        for col in self.condition_columns:
            if col not in df.columns:
                logger.warning(f"Condition column '{col}' not found in data. Creating empty column.")
                df[col] = "Not specified"
                
        return df
    
    def _replace_special_chars(self, text):
        """Replace special characters that might cause issues"""
        # Map of special characters to their replacements
        replacements = {
            'μ': 'u',       # micro
            '°': 'deg',     # degree
            'α': 'alpha',   # alpha
            'β': 'beta',    # beta
            'γ': 'gamma',   # gamma
            'δ': 'delta',   # delta
            '±': '+/-',     # plus-minus
            '≤': '<=',      # less than or equal
            '≥': '>=',      # greater than or equal
            '×': 'x',       # multiplication
            '÷': '/',       # division
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        return text

    def extract_valid_data(self):
        groups = self.df.groupby(self.condition_columns)
        group_sizes = groups.size()
        
        # Log group sizes instead of print
        logger.info("-"*25+"Number of instances"+"-"*25)
        logger.info(f"\n{group_sizes}")
        logger.info("-"*25+"-"*len("Number of instances")+"-"*25)
        
        enough_groups = group_sizes[group_sizes>=50].index
        valid_groups = []
        
        logger.info("Checking training quorum requirements for each test group:")
        
        for each_group in groups:
            group_name = " | ".join([str(x) for x in each_group[0]])
            
            if each_group[0] in enough_groups:
                task_type = recognize_task_type(each_group[1])
                
                if self.satisfy_training_quorum(each_group[1], task_type):
                    valid_groups.append(each_group)
                    logger.info(f"✓ Group '{group_name}' passed training quorum requirements for {task_type}")
                else:
                    logger.warning(f"✗ Group '{group_name}' failed to satisfy training quorum for {task_type}")
            else:
                logger.warning(f"✗ Group '{group_name}' has insufficient data points ({group_sizes[each_group[0]]} < 50)")
                
        logger.info(f"Total groups that meet training quorum: {len(valid_groups)}/{len(groups)}")
        return valid_groups

    def satisfy_training_quorum(self, each_df, task_type):
        if task_type.lower() == 'classification':        
            counts = each_df[self.activity_col].value_counts()
            active_count = counts.get('active', 0)
            inactive_count = counts.get('inactive', 0)
            if active_count < 25 or inactive_count < 25:
                logger.warning(f"Training quorum failed for classification. Active: {active_count}, Inactive: {inactive_count} (need at least 25 of each)")
                return False
            else:
                logger.info(f"Training quorum met for classification. Active: {active_count}, Inactive: {inactive_count}, Total: {active_count+inactive_count}")
                return True
        else:  # regression
            total_count = len(each_df)
            
            if 'Measurement_Relation' in each_df.columns:
                uncensored_count = len(each_df[~each_df['Measurement_Relation'].isin(['>', '<'])])
            else:
                has_relation_mask = each_df[self.activity_col].astype(str).str.contains('^[<>]')
                uncensored_count = len(each_df[~has_relation_mask])
                
            if total_count < 50 or uncensored_count < 25:
                logger.warning(f"Training quorum failed for regression. Total: {total_count}/50, Uncensored: {uncensored_count}/25")
                return False
            else:
                logger.info(f"Training quorum met for regression. Total: {total_count}/50, Uncensored: {uncensored_count}/25")
                return True
    
    def process(self):
        logger.info("Starting data group extraction process...")
        valid_groups = self.extract_valid_data()
        
        if len(valid_groups) == 0:
            logger.warning("No test groups met training quorum requirements. Please check your data.")
        else:
            logger.info(f"Extracted {len(valid_groups)} valid test groups that meet training quorum")
            
        return valid_groups


class Preprocessor:
    def __init__(self,
                 df:pd.DataFrame,
                 task_type:str,
                 task:str,
                 smiles_column:str='SMILES_Structure_Parent',
                 activity_column:str='Measurement_Value',
                 remove_salt:bool=True,
                 keep_stereo:bool=False, 
                 keep_duplicates:bool=False,
                 detect_outliers:bool=False,
                 scale_activity:bool=True,
                 threshold=None):
        # Load input data
        self.df = df
        self.activity_col = activity_column
        self.smiles_col = smiles_column
        self.check_task(task_type, task)
        # Check label
        self.threshold = threshold
        self.inspect_label()
        # Parameters for chemical preprocessing
        self.remove_salt = remove_salt
        self.keep_stereo = keep_stereo
        self.keep_duplicates = keep_duplicates
        self.scale_activity = scale_activity
        self.detect_outliers = detect_outliers
        self.active_is_high:bool = True
        self.final_cols = []
        self.smiles_column = smiles_column
        self.remover = SaltRemover.SaltRemover()
        self.uncharger = MolStandardize.rdMolStandardize.Uncharger()
        self.enumerator = MolStandardize.rdMolStandardize.TautomerEnumerator()
        
        # Add Test_Dose to final columns for Pharmacokinetics tests
        if 'Test' in self.df.columns and 'Pharmacokinetics' in self.df['Test'].values:
            if 'Test_Dose' in self.df.columns:
                logger.info("Pharmacokinetics test detected. Test_Dose column will be included in output.")
                self.final_cols.append('Test_Dose')
            else:
                logger.warning("Pharmacokinetics test detected but Test_Dose column not found. Adding with default value.")
                self.df['Test_Dose'] = "Unknown"
                self.final_cols.append('Test_Dose')
        
        # Check if Chemical_ID column exists
        self.has_chemical_id = False
        if 'Chemical ID' in self.df.columns:
            self.has_chemical_id = True
            self.final_cols.append('Chemical ID')

    def check_task(self, task_type, task=None):
        """
        Check if task type and task information are valid based on valid_options.csv
        """
        if task_type.lower() not in ['classification', 'regression']:
            raise ValueError(f"Task should be specified (Classification or Regression). Current task: {task_type}")
        self.task_type = task_type.lower()
        
        if task is not None:
            # Load valid options from csv
            options_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "valid_options.csv")
            
            try:
                if os.path.exists(options_file):
                    valid_options = pd.read_csv(options_file)
                    
                    # Get unique valid values for each column
                    valid_tests = set(valid_options['Test'].dropna())
                    valid_test_types = set(valid_options['Test_Type'].dropna())
                    valid_measurement_types = set(valid_options['Measurement_Type'].dropna())
                    
                    # Check task components based on type
                    task_components = []
                    
                    # Handle different types of task input
                    if isinstance(task, (list, tuple)):
                        # If task is already a tuple or list, use its elements directly
                        task_components = task
                    else:
                        # If task is a string, split by pipe character
                        task_components = str(task).split('|')
                    
                    # First component should be a Test value
                    if len(task_components) >= 1:
                        test_value = task_components[0]
                        if isinstance(test_value, str):
                            test_value = test_value.strip()
                        if test_value not in valid_tests and test_value != "Not specified":
                            logger.warning(f"Test '{test_value}' is not in the list of recognized tests.")
                    
                    # Second component should be a Test_Type value
                    if len(task_components) >= 2:
                        test_type_value = task_components[1]
                        if isinstance(test_type_value, str):
                            test_type_value = test_type_value.strip()
                        if test_type_value not in valid_test_types and test_type_value != "Not specified":
                            logger.warning(f"Test_Type '{test_type_value}' is not in the list of recognized test types.")
                    
                    # Fourth component (if exists) should be a Measurement_Type value
                    if len(task_components) >= 4:
                        measurement_type_value = task_components[3]
                        if isinstance(measurement_type_value, str):
                            measurement_type_value = measurement_type_value.strip()
                        if measurement_type_value not in valid_measurement_types and measurement_type_value != "Not specified":
                            logger.warning(f"Measurement_Type '{measurement_type_value}' is not in the list of recognized measurement types.")
                else:
                    logger.warning(f"Valid options file not found: {options_file}. Skipping validation.")
            except Exception as e:
                logger.warning(f"Error validating task against valid_options.csv: {e}. Continuing without validation.")
            
            self.task = task
        
    def inspect_label(self):
        unique_labels = self.df[self.activity_col].dropna().unique()
        num_unique = len(unique_labels)
        if num_unique == 2:
            if set(unique_labels) <= {0, 1} or set(unique_labels) <= {True, False}:
                label_type = 'Binary'
            else:
                label_type = 'Categorical'
        elif num_unique < self.df.shape[0]/2:
            label_type = 'Categorical'
        else:
            label_type = 'Continuous'
        if label_type=="Binary" and self.task_type=="regression":
            raise ValueError(f"Regression is not available for binary label")
        elif label_type=="Categorical" and self.task_type=="regression":
            raise ValueError(f"Regression is not available for categorical label")
        elif label_type=='Continuous' and self.task_type=="classification" and self.threshold is None:
            raise ValueError(f"To perform classification on continuous label, threshold must be given.")
        self.label_type = label_type
    
    def detect_outlier_from_distribution(self):
        activity_data = self.df[self.activity_col].dropna()
        stat, p_value = normaltest(activity_data)
        
        if p_value >= 0.05:
            print("The data follows a normal distribution. Applying statistical outlier detection.")
            self.outliers = self.detect_outliers_statistical()
        else:
            print("The data does not follow a normal distribution. Applying alternative methods.")
            if self.task_type=="regression":
                self.outliers = self.detect_outliers_density_based()
            else:
                self.outliers = self.detect_outliers_classification_based()

    def inspect_compound_id(self):
        cols = [each.lower() for each in self.df.columns]
        if 'Chemical ID' in cols:
            self.is_compound_id = True
            
    def preprocess_compound(self, smiles:str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"SMILES string is not valid: {smiles}")

        if self.remove_salt:                                # Remove salt
            mol = self.remover.StripMol(mol)
        
        for atom in mol.GetAtoms():                         # Remove isotope
            atom.SetIsotope(0)

        mol = self.uncharger.uncharge(mol)                  # Uncharge
        Chem.SetAromaticity(mol)                            # Aromaticity
        mol = Chem.RemoveHs(mol)                            # Remove H

        if not self.keep_stereo:                            # Stereochemistry
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        
        mol = self.enumerator.Canonicalize(mol)             # Tautomer
        
        try:
            Chem.SanitizeMol(mol)                           # Check valence error
        except Chem.rdChem.KekulizeException:
            raise ValueError(f"Valence error detected in molecule: {smiles}")
        
        standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)    # Scaffold extraction
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        return standardized_smiles, scaffold_smiles
    
    def preprocess_compounds(self, smiles_list: Iterable[str]):
        standardized_smiles_list = []
        scaffolds = []
        
        if len(smiles_list) < 1000 or multiprocessing.cpu_count() <= 2:
            logger.info("Processing compounds in single-process mode.")
            for smiles in smiles_list:
                try:
                    standardized_smiles, scaffold_smiles = self.preprocess_compound(smiles)
                    standardized_smiles_list.append(standardized_smiles)
                    scaffolds.append(scaffold_smiles)
                except ValueError as e:
                    logger.warning(f"Error processing SMILES {smiles}: {e}")
                    standardized_smiles_list.append(None)
                    scaffolds.append(None)

        else:
            logger.info(f"Processing compounds in parallel with {multiprocessing.cpu_count()} cores.")
            smiles_list = list(smiles_list)  # Ensure it's a list
            
            def process_smiles_batch(smiles_batch):
                results = []
                for smiles in smiles_batch:
                    try:
                        std_smiles, scaffold = self.preprocess_compound(smiles)
                        results.append((std_smiles, scaffold))
                    except ValueError as e:
                        logger.warning(f"Error processing SMILES {smiles}: {e}")
                        results.append((None, None))
                return results
                
            batch_size = max(100, len(smiles_list) // (multiprocessing.cpu_count() * 2))
            smiles_batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]

            start_time = time.time()
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(process_smiles_batch, smiles_batches)

            flat_results = [item for sublist in results for item in sublist]
            standardized_smiles_list = [item[0] for item in flat_results]
            scaffolds = [item[1] for item in flat_results]
            
            logger.info(f"Parallel processing completed in {time.time() - start_time:.2f} seconds")
            
        self.df["Standardized_SMILES"] = standardized_smiles_list
        self.df["Scaffold"] = scaffolds
        self.final_cols+=["Standardized_SMILES", "Scaffold"]
    
    def detect_outliers_statistical(self):
        q1 = self.df[self.activity_col].quantile(0.25)
        q3 = self.df[self.activity_col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = self.df[(self.df[self.activity_col] < lower_bound)|(self.df[self.activity_col] > upper_bound)]
        print(f"Found {len(outliers)} outliers using IQR method.")
        return outliers

    def detect_outliers_density_based(self):
        activity_data = self.df[[self.activity_col]].dropna()
        lof = LocalOutlierFactor(n_neighbors=20)
        labels = lof.fit_predict(activity_data)
        outliers = self.df[labels == -1]
        print(f"Found {len(outliers)} outliers using LOF.")
        return outliers

    def detect_outliers_classification_based(self):
        activity_data = self.df[[self.activity_col]].dropna()
        svm = OneClassSVM(kernel='rbf', gamma='auto')
        labels = svm.fit_predict(activity_data)
        outliers = self.df[labels == -1]
        print(f"Found {len(outliers)} outliers using OneClassSVM.")
        return outliers
    
    def detect_outliers_model_based(self):
        activity_data = self.df[[self.activity_col]].dropna()
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(activity_data)
        scores = gmm.score_samples(activity_data)
        threshold = np.percentile(scores, 5)
        outliers = self.df[scores < threshold]
        print(f"Found {len(outliers)} outliers using Gaussian Mixture.")
        return outliers
        
    def scale_experiment_values(self, labels: Iterable):
        """
        Scale experimental values.
        """
        # Scale only numeric data
        try:
            # Filter numeric data only
            numeric_mask = pd.to_numeric(labels, errors='coerce').notna()
            if numeric_mask.sum() > 0:
                numeric_values = pd.to_numeric(labels[numeric_mask])
                
                if self.task_type=="regression" and self.scale_activity:
                # Apply scaling
                    scaler = StandardScaler()
                    scaled_values = scaler.fit_transform(numeric_values.values.reshape(-1, 1))
                    
                    # Add scaled values to the original dataframe
                    scaled_col = self.activity_col + "_scaled"
                    self.df[scaled_col] = np.nan
                    self.df.loc[numeric_mask, scaled_col] = scaled_values.flatten()
                    self.final_cols.append(scaled_col)
                logger.info(f"Scaled {numeric_mask.sum()} numeric values")
            else:
                logger.warning("No numeric values found for scaling")
        except Exception as e:
            logger.warning(f"Error scaling values: {e}")
    
    def create_classification_label(self, labels: Iterable):
        """
        Convert numeric data to classification labels.
        """
        try:
            # If already classification labels (active/inactive)
            if set(labels.unique()) <= {'active', 'inactive', 'Active', 'Inactive'}:
                # Standardize case
                self.df["Classification_label"] = labels.str.lower()
                self.final_cols.append("Classification_label")
                logger.info("Used existing classification labels")
                return
                
            # Filter numeric data only
            numeric_mask = pd.to_numeric(labels, errors='coerce').notna()
            if numeric_mask.sum() > 0 and self.threshold is not None:
                # Convert numeric data to classification labels
                numeric_values = pd.to_numeric(labels[numeric_mask])
                
                # Threshold-based classification
                self.df["Classification_label"] = "Not classified"
                if self.active_is_high:
                    self.df.loc[numeric_mask, "Classification_label"] = np.where(
                        numeric_values >= self.threshold, 'active', 'inactive')
                else:
                    self.df.loc[numeric_mask, "Classification_label"] = np.where(
                        numeric_values <= self.threshold, 'active', 'inactive')
                
                self.final_cols.append("Classification_label")
                logger.info(f"Created classification labels using threshold {self.threshold}")
            else:
                logger.warning("Could not create classification labels: missing threshold or no numeric data")
        except Exception as e:
            logger.warning(f"Error creating classification labels: {e}")

    def preprocess_labels(self, labels: Iterable):
        """
        Preprocess labels.
        """
        # Try string -> numeric conversion
        try:
            # Process classification values like 'active'/'inactive'
            if self.label_type == 'Categorical' and self.task_type == 'classification':
                # Standardize string labels
                if 'Classification_label' not in self.df.columns:
                    self.create_classification_label(labels)
            
            # Process numeric data
            if self.detect_outliers:
                self.detect_outlier_from_distribution()
                
            # Apply scaling (numeric only)
            if np.issubdtype(labels.dtype, np.number) or labels.str.match(r'^[<>]?\s*\d+\.?\d*$').any():
                self.scale_experiment_values(labels)
            
            # Convert continuous values -> classification (using threshold)
            if self.label_type == 'Continuous' and self.task_type == "classification" and self.threshold is not None:
                self.create_classification_label(labels)
                
        except Exception as e:
            logger.error(f"Error preprocessing labels: {e}")
            # Add the base column to the final column list
            if self.activity_col not in self.final_cols:
                self.final_cols.append(self.activity_col)

    def preprocess(self):
        compounds = self.df[self.smiles_col]
        labels = self.df[self.activity_col]
        self.preprocess_compounds(compounds)
        self.preprocess_labels(labels)
        if self.detect_outliers:
            self.detect_outlier_from_distribution()
            self.df.drop(self.outliers.index, inplace=True)
            
        # Improved duplicate removal logic
        # Skip duplicate removal for Pharmacokinetics data
        is_pk_data = False
        # Check either Test or Test_Type columns for Pharmacokinetics
        if 'Test' in self.df.columns and 'Pharmacokinetics' in self.df['Test'].values:
            is_pk_data = True
            logger.info("Pharmacokinetics data detected. Skipping duplicate removal to preserve multiple measurements.")
        
        if not self.keep_duplicates and not is_pk_data:
            # Record data count before duplicate removal
            before_count = len(self.df)
            
            # Remove duplicates (remove all duplicates with keep=False)
            self.df.drop_duplicates(subset=["Standardized_SMILES"], keep=False, inplace=True, ignore_index=True)
            
            # Record data count after duplicate removal
            after_count = len(self.df)
            if after_count == 0 and before_count > 0:
                logger.warning(f"All data removed during duplicate removal! Original count: {before_count}")
                # Try again with keep='first' to prevent complete data removal
                self.df = pd.DataFrame(compounds).reset_index(drop=True)
                self.df[self.activity_col] = labels.reset_index(drop=True)
                self.preprocess_compounds(compounds)
                self.preprocess_labels(labels)
                self.df.drop_duplicates(subset=["Standardized_SMILES"], keep='first', inplace=True, ignore_index=True)
                logger.info(f"Recovered {len(self.df)} records using keep='first' strategy")
            else:
                logger.info(f"Removed {before_count - after_count} duplicate records. Remaining: {after_count}")
            
        # Add basic columns to ensure they're included
        for col in [self.activity_col, 'Test', 'Test_Type', 'Test_Subject', 'Measurement_Type']:
            if col in self.df.columns and col not in self.final_cols:
                self.final_cols.append(col)
        
        # Check if result is empty
        if len(self.df) == 0:
            logger.error("Preprocessing resulted in empty dataset!")
        else:
            logger.info(f"Preprocessing completed. Final dataset contains {len(self.df)} records.")
                
        return self.df[self.final_cols]

class DataVisualizer:
    """
    Class for data and chemical structure visualization
    """
    def __init__(self, df, smiles_col='Standardized_SMILES', activity_col='Measurement_Value', scale_activity=True):
        self.df = df
        self.smiles_col = smiles_col
        self.activity_col = f"{activity_col}_scaled" if scale_activity and f"{activity_col}_scaled" in df.columns else activity_col
        self.has_chemical_id = 'Chemical ID' in df.columns
    
    def visualize_molecules(self, n_mols=10, output_path=None):
        """
        Visualize molecular structures
        
        Parameters:
        -----------
        n_mols : int
            Number of molecules to visualize
        output_path : str
            Path to save the output image
        """
        
        # Check if we have molecules to visualize
        if self.smiles_col not in self.df.columns:
            logger.error(f"SMILES column '{self.smiles_col}' not found in data")
            return None
            
        # Filter valid SMILES - 중복도 처리
        valid_smiles_df = self.df[self.df[self.smiles_col].notna()].copy()
        if len(valid_smiles_df) == 0:
            logger.warning("No valid SMILES found for visualization")
            return None
            
        # 중복된 SMILES가 있을 경우를 처리
        if len(valid_smiles_df) > len(valid_smiles_df[self.smiles_col].unique()):
            logger.info(f"Found {len(valid_smiles_df) - len(valid_smiles_df[self.smiles_col].unique())} duplicate SMILES. Keeping first occurrence of each.")
            # 각 고유 SMILES의 첫 번째 행만 유지
            valid_smiles_df = valid_smiles_df.drop_duplicates(subset=[self.smiles_col], keep='first')
        
        # 표시할 분자 수 제한
        valid_smiles_df = valid_smiles_df.head(n_mols)
            
        # Determine grid dimensions
        n_valid = len(valid_smiles_df)
        n_cols = min(3, n_valid)  # Maximum 3 molecules per row
        n_rows = (n_valid + n_cols - 1) // n_cols  # Ceiling division
        
        try:
            # Create a matplotlib figure for custom layout
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
            
            # Handle single subplot case
            if n_rows * n_cols == 1:
                axes = np.array([axes])
            
            # Flatten axes array for easier iteration
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
            
            # Iterate through valid molecules
            for i, (idx, row) in enumerate(valid_smiles_df.iterrows()):
                if i >= len(axes):
                    break
                    
                # Convert SMILES to molecule
                mol = Chem.MolFromSmiles(row[self.smiles_col])
                if mol is None:
                    logger.warning(f"Failed to convert SMILES to molecule: {row[self.smiles_col]}")
                    continue
                
                # Create molecule image
                img = Draw.MolToImage(mol, size=(300, 300))
                
                # Display molecule in the current axis
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Create title with activity value
                title_parts = []
                
                # Add activity value if available
                if pd.notna(row.get(self.activity_col)):
                    title_parts.append(f"Activity: {row.get(self.activity_col):.2f}")
                
                title_parts.append(f"\n")
                
                # Add Chemical ID on a separate line if available
                if self.has_chemical_id and pd.notna(row.get('Chemical ID')):
                    title_parts.append(f"ID: {row.get('Chemical ID')}")
                
                # Set title with a larger fontsize
                axes[i].set_title("\n".join(title_parts), fontsize=12)
            
            # Hide unused subplots
            for j in range(min(i+1, len(axes)), len(axes)):
                axes[j].axis('off')
                axes[j].set_visible(False)
            
            # Adjust layout
            plt.tight_layout(pad=2.0)
            
            # Save or return the figure
            if output_path:
                plt.savefig(output_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Custom molecule visualization saved to {output_path}")
                return None
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error creating custom molecule visualization: {e}")
            
            # Fallback to original RDKit method if the custom method fails
            try:
                # Generate molecules and legends
                mols = []
                legends = []
                
                for idx, row in valid_smiles_df.iterrows():
                    try:
                        mol = Chem.MolFromSmiles(row[self.smiles_col])
                        if mol:
                            mols.append(mol)
                            
                            # Create two-line legend
                            legend_parts = []
                            if pd.notna(row.get(self.activity_col)):
                                legend_parts.append(f"{row.get(self.activity_col):.2f}")
                            
                            if self.has_chemical_id and pd.notna(row.get('Chemical ID')):
                                if legend_parts:
                                    legend_parts.append("\n\n\n")
                                legend_parts.append(f"{row.get('Chemical ID')}")
                                
                            legends.append("".join(legend_parts))
                    except Exception as mol_error:
                        logger.warning(f"Error processing molecule: {mol_error}")
                
                if not mols:
                    logger.warning("No valid molecules to visualize")
                    return None
                    
                img = Draw.MolsToGridImage(
                    mols, 
                    molsPerRow=min(3, len(mols)),
                    subImgSize=(300, 300),
                    legends=legends,
                    maxMols=n_mols
                )
                
                if output_path:
                    img.save(output_path)
                    logger.info(f"Fallback molecule visualization saved to {output_path}")
                    return None
                else:
                    return img
                    
            except Exception as e2:
                logger.error(f"Both visualization methods failed: {e2}")
                return None
    
    def plot_activity_distribution(self, output_path=None):
        """
        Visualize activity value distribution
        
        Parameters:
        -----------
        output_path : str
            Path to save the output image
        """
        
        if self.activity_col not in self.df.columns:
            logger.error(f"Activity column '{self.activity_col}' not found in dataframe")
            return
        
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[self.activity_col].dropna(), kde=True)
        plt.title(f'Distribution of {self.activity_col}')
        plt.xlabel(self.activity_col)
        plt.ylabel('Frequency')
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Activity distribution plot saved to {output_path}")
        else:
            return plt.gcf()
    
    def plot_scaffold_diversity(self, top_n=10, output_path=None):
        """
        Visualize scaffold diversity
        
        Parameters:
        -----------
        top_n : int
            Number of top scaffolds to display
        output_path : str
            Path to save the output image
        """
        
        if 'Scaffold' not in self.df.columns:
            logger.error("Scaffold column not found in dataframe")
            return
        
        # Calculate scaffold frequency
        scaffold_counts = self.df['Scaffold'].value_counts().head(top_n)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=scaffold_counts.index, y=scaffold_counts.values)
        plt.title(f'Top {top_n} Scaffolds')
        plt.xlabel('Scaffold')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Scaffold diversity plot saved to {output_path}")
        else:
            return plt.gcf()

class DataSplitter:
    """
    Class for splitting data for machine learning
    """
    def __init__(self, df, smiles_col='Standardized_SMILES', scaffold_col='Scaffold', 
                 activity_col='Measurement_Value', random_state=42):
        self.df = df
        self.smiles_col = smiles_col
        self.scaffold_col = scaffold_col
        self.activity_col = activity_col
        self.random_state = random_state
    
    def random_split(self, test_size=0.2, valid_size=0.1):
        """
        Create train/validation/test sets using random split
        
        Parameters:
        -----------
        test_size : float
            Test set ratio
        valid_size : float
            Validation set ratio
            
        Returns:
        --------
        dict : Dictionary with 'train', 'valid', 'test' dataframe keys
        """
        # First split: train+valid vs test
        train_valid, test = train_test_split(
            self.df, test_size=test_size, random_state=self.random_state
        )
        
        # Second split: train vs valid
        # valid_size is adjusted based on train_valid, not the original dataset
        if valid_size > 0:
            valid_adjusted = valid_size / (1 - test_size)
            train, valid = train_test_split(
                train_valid, test_size=valid_adjusted, random_state=self.random_state
            )
            return {'train': train, 'valid': valid, 'test': test}
        else:
            return {'train': train_valid, 'test': test}
    
    def scaffold_split(self, test_size=0.2, valid_size=0.1):
        """
        Create train/validation/test sets using scaffold-based split
        Prevents compounds with similar chemical structures from appearing in different sets
        
        Parameters:
        -----------
        test_size : float
            Test set ratio
        valid_size : float
            Validation set ratio
            
        Returns:
        --------
        dict : Dictionary with 'train', 'valid', 'test' dataframe keys
        """
        if self.scaffold_col not in self.df.columns:
            logger.warning("Scaffold column not found. Falling back to random split.")
            return self.random_split(test_size, valid_size)
        
        # Create scaffold groups
        scaffold_groups = {}
        for idx, row in self.df.iterrows():
            scaffold = row[self.scaffold_col]
            if scaffold not in scaffold_groups:
                scaffold_groups[scaffold] = []
            scaffold_groups[scaffold].append(idx)
        
        # Sort scaffolds by size (largest first)
        scaffolds_sorted = sorted(scaffold_groups.keys(), 
                                key=lambda x: len(scaffold_groups[x]), 
                                reverse=True)
        
        # Initialize index arrays
        train_indices = []
        valid_indices = []
        test_indices = []
        
        # Split data by scaffold size
        for scaffold in scaffolds_sorted:
            indices = scaffold_groups[scaffold]
            
            # Add to test set if test target not reached
            if len(test_indices) / len(self.df) < test_size:
                test_indices.extend(indices)
            # Add to validation set if validation target not reached
            elif valid_size > 0 and len(valid_indices) / len(self.df) < valid_size:
                valid_indices.extend(indices)
            # Add the rest to training set
            else:
                train_indices.extend(indices)
        
        # Create result
        result = {
            'train': self.df.loc[train_indices].reset_index(drop=True)
        }
        
        if valid_size > 0:
            result['valid'] = self.df.loc[valid_indices].reset_index(drop=True)
        
        result['test'] = self.df.loc[test_indices].reset_index(drop=True)
        
        # Log results
        logger.info(f"Scaffold split: train={len(train_indices)}, "
                  f"{'valid=' + str(len(valid_indices)) + ', ' if valid_size > 0 else ''}"
                  f"test={len(test_indices)}")
        
        return result
    
    def stratified_split(self, test_size=0.2, valid_size=0.1, bins=10):
        """
        Create train/validation/test sets using stratified split
        Maintains similar activity value distribution across all sets
        
        Parameters:
        -----------
        test_size : float
            Test set ratio
        valid_size : float
            Validation set ratio
        bins : int
            Number of bins for classifying continuous activity values
            
        Returns:
        --------
        dict : Dictionary with 'train', 'valid', 'test' dataframe keys
        """
        if self.activity_col not in self.df.columns:
            logger.warning("Activity column not found. Falling back to random split.")
            return self.random_split(test_size, valid_size)
        
        # Convert continuous values to categorical for stratified sampling
        if np.issubdtype(self.df[self.activity_col].dtype, np.number):
            # Divide continuous variable into bins
            activity_bins = pd.qcut(self.df[self.activity_col], bins, 
                                  labels=False, duplicates='drop')
        else:
            # Use as-is if already categorical
            activity_bins = self.df[self.activity_col]
        
        # First split: train+valid vs test
        train_valid, test = train_test_split(
            self.df, test_size=test_size, 
            stratify=activity_bins, random_state=self.random_state
        )
        
        # Second split: train vs valid
        if valid_size > 0:
            # Stratify in the same way as the first split
            if np.issubdtype(self.df[self.activity_col].dtype, np.number):
                train_valid_bins = pd.qcut(train_valid[self.activity_col], bins, 
                                        labels=False, duplicates='drop')
            else:
                train_valid_bins = train_valid[self.activity_col]
            
            valid_adjusted = valid_size / (1 - test_size)
            train, valid = train_test_split(
                train_valid, test_size=valid_adjusted, 
                stratify=train_valid_bins, random_state=self.random_state
            )
            return {'train': train, 'valid': valid, 'test': test}
        else:
            return {'train': train_valid, 'test': test}
    
    def split(self, method='random', test_size=0.2, valid_size=0.1):
        """
        Split data using the specified method
        
        Parameters:
        -----------
        method : str
            One of 'random', 'scaffold', 'stratified'
        test_size : float
            Test set ratio
        valid_size : float
            Validation set ratio
            
        Returns:
        --------
        dict : Data set dictionary
        """
        if method == 'scaffold':
            return self.scaffold_split(test_size, valid_size)
        elif method == 'stratified':
            return self.stratified_split(test_size, valid_size)
        else:  # random
            return self.random_split(test_size, valid_size)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./processed_data", required=False)
    parser.add_argument("--visualize", type=bool, default=False, help="Generate visualizations")
    parser.add_argument("--parallel", type=bool, default=False, help="Use parallel processing")
    parser.add_argument("--scale_activity", type=bool, default=True, help="Scale activity values")
    parser.add_argument("--split", choices=["random", "scaffold", "stratified"], 
                      help="Split data for machine learning")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set size")
    parser.add_argument("--valid_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--activity_col", type=str, default="Measurement_Value", 
                     help="Column name for activity values")
    parser.add_argument("--smiles_col", type=str, default="SMILES_Structure_Parent",
                     help="Column name for SMILES structures")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    # Set up log directory inside output directory
    log_dir = os.path.join(args.output_path, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging with file handler
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"preprocess_{timestamp}.log")
    
    # Set up logging level
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    if root_logger.handlers:
        for handler in root_logger.handlers:
            root_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logger.info(f"Logging to file: {log_file}")
    
    start_time = time.time()
    logger.info(f"Starting processing of {args.input_path}")
    
    try:
        # Check if input file exists
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input file not found: {args.input_path}")
        
        # Inspect data
        inspector = DataInspector(
            input_path=args.input_path,
            smiles_column=args.smiles_col,
            activity_column=args.activity_col
        )
        
        valid_groups = inspector.process()
        
        if len(valid_groups) < 1:
            logger.error("No test met training quorum. Please check your data.")
            raise TrainingQuorumError("No test met training quorum")
        
        # Create summary of valid groups
        logger.info("-"*50)
        logger.info("SUMMARY OF VALID TEST GROUPS:")
        for i, each_group in enumerate(valid_groups):
            group_name = " | ".join([str(x) for x in each_group[0]])
            task_type = recognize_task_type(each_group[1], activity_col=args.activity_col)
            logger.info(f"{i+1}. {group_name} - Type: {task_type} - Samples: {len(each_group[1])}")
        logger.info("-"*50)
        
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
            logger.info(f"Created output directory: {args.output_path}")
        
        for i, each_group in enumerate(valid_groups):
            task = each_group[0]
            each_df = each_group[1]
            task_type = recognize_task_type(each_df, activity_col=args.activity_col)
            logger.info(f"Processing group {i+1}/{len(valid_groups)}: {task} ({task_type})")
            
            try:
                preprocessor = Preprocessor(
                    each_df, 
                    task_type, 
                    task,
                    smiles_column=args.smiles_col,
                    activity_column=args.activity_col,
                    scale_activity=args.scale_activity
                )
                processed_data = preprocessor.preprocess()
                
                # Check if result is empty
                if processed_data.empty:
                    logger.warning(f"Preprocessing for group {task} resulted in empty dataset. Skipping output.")
                    continue
                
                # Improved file naming logic
                base_name = os.path.basename(args.input_path).split('.')[0]  # Remove extension
                
                # Convert task to string if it's a tuple or list
                task_str = ""
                if isinstance(task, (tuple, list)):
                    # Join tuple/list elements with underscores
                    task_str = "_".join([str(x).replace(' ', '_') for x in task])
                else:
                    # Use as-is for strings, replacing spaces with underscores
                    task_str = str(task).replace(' ', '_')
                
                # Remove invalid characters for filenames
                task_str = re.sub(r"([^\w\-\.]),", '_', task_str)
                
                output_file = f'{args.output_path}/{base_name}_{task_str}_processed.csv'
                processed_data.to_csv(output_file, index=False)
                logger.info(f"Saved processed data to {output_file}")
                logger.info(f"Processed data contains {len(processed_data)} records with {len(processed_data.columns)} columns.")
                
                if args.split and not processed_data.empty:
                    splitter = DataSplitter(
                        processed_data,
                        smiles_col='Standardized_SMILES',
                        activity_col=args.activity_col
                    )
                    split_data = splitter.split(
                        method=args.split, 
                        test_size=args.test_size,
                        valid_size=args.valid_size
                    )
                    
                    split_dir = f"{args.output_path}/splits"
                    if not os.path.exists(split_dir):
                        os.makedirs(split_dir)
                    
                    for split_name, split_df in split_data.items():
                        if not split_df.empty:
                            split_file = f"{split_dir}/{base_name}_{task_str}_{split_name}.csv"
                            split_df.to_csv(split_file, index=False)
                            logger.info(f"Saved {split_name} split to {split_file} ({len(split_df)} samples)")
                
                if args.visualize and not processed_data.empty:
                    viz_dir = f"{args.output_path}/visualizations"
                    if not os.path.exists(viz_dir):
                        os.makedirs(viz_dir)
                    
                    visualizer = DataVisualizer(
                        processed_data, 
                        smiles_col='Standardized_SMILES',
                        activity_col=args.activity_col,
                        scale_activity=args.scale_activity
                    )
                    
                    try:
                        mol_viz_path = f"{viz_dir}/{base_name}_{task_str}_molecules.png"
                        visualizer.visualize_molecules(n_mols=10, output_path=mol_viz_path)
                        
                        act_dist_path = f"{viz_dir}/{base_name}_{task_str}_activity_dist.png"
                        visualizer.plot_activity_distribution(output_path=act_dist_path)
                        
                        scaffold_path = f"{viz_dir}/{base_name}_{task_str}_scaffolds.png"
                        visualizer.plot_scaffold_diversity(output_path=scaffold_path)
                    except Exception as viz_error:
                        logger.error(f"Error during visualization: {viz_error}")
                        
            except Exception as process_error:
                logger.error(f"Error processing group {task}: {process_error}")
                continue
        
        logger.info(f"All processing completed successfully in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())