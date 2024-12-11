import pandas as pd
import numpy as np
from typing import Iterable
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
# from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from scipy.stats import normaltest


def recognize_task_type(each_df, activity_col='Measurement_Value'):
    if 'Not specified' in each_df[activity_col]:
        return "classification"
    else:
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
                 condition_columns:list=['Test', 'Test_Type', 'Test_Subject', 'Measurment_Type']
                 ):
        self.smiles_col = smiles_column
        self.activity_col = activity_column
        self.condition_columns = condition_columns
        self.df = self.load_data(input_path)
    
    def load_data(self, input_path):
        if not input_path.endwith('.csv'):
            raise ValueError(f"Input file must be a CSV file. Current file is: {input_path}")
        self.df = pd.read_csv(input_path).fillna("Not specified")
        if self.smiles_col not in self.df.columns:
            raise ValueError(f"SMILES column '{self.smiles_col}' not found in data.")
        if not np.issubdtype(self.df[self.activity_col].dtype, np.number):
            raise ValueError(f"Activity column '{self.activity_col}' must contain numeric values.")

    def extract_valid_data(self):
        groups = self.df.groupby(self.condition_columns)
        group_sizes = groups.size()
        print("-"*25+"Number of instances"+"-"*25)
        print(group_sizes)
        print("-"*25+"-"*len("Number of instances")+"-"*25)
        enough_groups = group_sizes[group_sizes>=50].index
        valid_groups = []
        for each_group in groups:
            if each_group[0] in enough_groups:
                if self.check_training_quorum(each_group[1]):
                    valid_groups.append(each_group)
            else:
                print(f"{each_group} failed to meet training quorum.")
                # task_type = recognize_task_type(each_group[1])
                # raise TrainingQuorumError(task_type)
        return valid_groups

    def satisfy_training_quorum(self, each_df, task_type):
        if task_type.lower() == 'classification':        
            counts = each_df[self.activity_col].value_counts()
            active_count = counts.get('active', 0)
            inactive_count = counts.get('inactive', 0)
            if active_count < 25 or inactive_count < 25:
                print(f"Training quorum failed for regression. Number of instances: {active_count}, {inactive_count}")
                # raise TrainingQuorumError(task_type)
            else:
                print(f"Training quorum met for classification. Number of instances: {active_count+inactive_count}")
                return True
        else:
            total_count = len(each_df)
            uncensored_count = len(each_df[~each_df['Measurement_Relation'].isin(['>', '<'])])
            if total_count < 50 or uncensored_count < 25:
                print(f"Training quorum failed for regression. Number of instances: {total_count}")
                # raise TrainingQuorumError(task_type)
            else:
                print(f"Training quorum met for regression. Number of instances: {total_count}")
                return True
    
    def process(self):
        valid_groups = self.extract_valid_data()
        final_groups = []
        for each_group in valid_groups:
            task_type = recognize_task_type(each_group[1])
            if self.satisfy_training_quorum(each_group[1], task_type):
                final_groups.append(each_group)
        return final_groups
    

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
        self.detect_outliers = detect_outliers
        self.active_is_high:bool = True
        self.final_cols = []
        self.smiles_column = smiles_column
        self.remover = SaltRemover.SaltRemover()
        self.uncharger = MolStandardize.rdMolStandardize.Uncharger()
        self.enumerator = MolStandardize.rdMolStandardize.TautomerEnumerator()

    def check_task(self, task_type):
        if task_type.lower() not in ['classification', 'regression']:
            raise ValueError(f"Task should be specified (Classification or Regression). Current task: {task_type}")
        self.task_type = task_type.lower()
        
    def inspect_label(self):
        unique_labels = self.df[self.activity_column].dropna().unique()
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
        activity_data = self.data[self.activity_col].dropna()
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
        if 'Chemical_ID' in cols:
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
        for smiles in smiles_list:
            standardized_smiles, scaffold_smiles = self.preprocess_compound(smiles)
            standardized_smiles_list.append(standardized_smiles)
            scaffolds.append(scaffold_smiles)
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
        
    def scale_experiment_values(self, labels: Iterable[float]):
        scaler = StandardScaler() # Should be downloaded from the central server at the final stage
        scaled_labels = scaler.fit_transform(labels.values.reshape(-1, 1)) # Should be converted to the scaler.transform
        self.df[self.activity_col+"_scaled"] = scaled_labels
        self.final_cols.append(self.activity_col+"_scaled")
    
    def create_classification_label(self, labels: Iterable[float]):
        if self.active_is_high:
            self.df["Classification_label"] = np.where(labels <= self.threshold, 'Active', 'Inactive')
        else:
            self.df["Classification_label"] = np.where(labels >= self.threshold, 'Active', 'Inactive')
        self.final_cols.append("Classification_label")

    def preprocess_labels(self, labels: Iterable[float]):
        if self.detect_outliers:
            self.detect_outlier_from_distribution(labels)
        self.scale_experiment_values(labels)
        if self.label_type=='Continuous' and self.task_type=="classification":
            self.create_classification_label(labels)

    def preprocess(self):
        compounds = self.df[self.smiles_col]
        labels = self.df[self.activity_col]
        self.preprocess_compounds(compounds)
        self.preprocess_labels(labels)
        if self.detect_outliers:
            self.detect_outlier_from_distribution
            self.df.drop(self.outliers.index, inplace=True)
        if not self.keep_duplicates:
            self.df.drop_duplicates(subset=["Standardized_SMILES"], keep=False, inplace=True, ignore_index=True)
        return self.df[self.final_cols]


if __name__ == "__main__":
    input_path = "./data_sample_2.csv"
    inspector = DataInspector(input_path=input_path)
    valid_groups = inspector.process()
    if len(valid_groups)<1:
        raise TrainingQuorumError("No test met training quorum")
    for each_group in valid_groups:
        task = each_group[0]
        each_df = each_group[1]
        task_type = recognize_task_type(each_df)
        preprocessor = Preprocessor(each_df, task_type, task)
        processed_data = preprocessor.preprocess()
        processed_data.to_csv(f'{input_path.rstrip('.csv')}_{task}_processed.csv')

"""
        # Check task
        self.main_tasks = [
            'solubility', 'caco-2', 'pampa', 'ppb', 'p-gp',
            'microsomal-stability', 'cyp1a2 inhibition', 'cyp2c9 inhibition',
            'cyp2c19 inhibition', 'cyp2d6 inhibition', 'cyp3a inhibition',
            'herg inhibition', 'ames test'
        ]
        if task.lower() not in self.main_tasks:
            raise ValueError(f"Task should be one of 'solubility', 'caco-2', 'pampa', 'ppb', 'p-gp', 'microsomal-stability', 'cyp1a2 inhibition', 'cyp2c9 inhibition', 'cyp2c19 inhibition', 'cyp2d6 inhibition', 'cyp3a inhibition', 'herg inhibition', 'ames test'")
"""