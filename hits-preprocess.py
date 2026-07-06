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
import textwrap

# Set pandas display options
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect display width
pd.set_option('display.max_colwidth', None)  # Show full content of each column
pd.set_option('display.expand_frame_repr', False)  # Don't wrap wide frames

import sys
import json
from typing import Iterable, List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig, MISSING
from omegaconf.errors import ConfigKeyError, ValidationError
from rdkit import Chem
from rdkit.Chem import SaltRemover, MolStandardize, Draw, Descriptors
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

# Unit conversion imports
try:
    import pint
    from pint import UnitRegistry
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False
    logger.warning("pint package not available. Unit conversion will be disabled.")

# The logger will be configured with file handler in main


@dataclass
class PreprocessConfig:
    """
    Structured configuration for the preprocessing pipeline.
    OmegaConf validates types and keeps attribute-style access.
    """
    input_path: str = MISSING
    output_path: str = "./processed_data"
    to_gist_matrix: bool = False
    gist_output: Optional[str] = None
    visualize: bool = False
    parallel: bool = False
    scale_activity: bool = True
    convert_units: bool = True
    correct_pH: bool = False
    pH_method: str = "all"
    target_pH: float = 7.4
    endpoint_mapper: str = "llm"
    manual_mapping_path: Optional[str] = None
    manual_min_similarity: float = 0.55
    manual_prefer_exact: bool = True
    split: Optional[str] = None
    test_size: float = 0.2
    valid_size: float = 0.1
    activity_col: str = "Measurement_Value"
    smiles_col: str = "SMILES_Structure_Parent"
    debug: bool = False
    config: Optional[str] = None


def _convert_argparse_style_to_cli(argv: List[str]) -> List[str]:
    """
    Convert traditional argparse style flags (e.g. --flag value) to OmegaConf overrides.
    This keeps backward compatibility with existing shell scripts.
    """
    converted: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token.startswith("--"):
            key = token.lstrip("-")
            if "=" in key:
                key_segment, raw_value = key.split("=", 1)
                converted.append(f"{key_segment.replace('-', '_')}={raw_value}")
                i += 1
                continue

            normalized_key = key.replace("-", "_")
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                value = argv[i + 1]
                i += 2
            else:
                value = "true"
                i += 1
            converted.append(f"{normalized_key}={value}")
        else:
            converted.append(token)
            i += 1
    return converted


def _print_help() -> None:
    help_text = textwrap.dedent(
        """
        Usage:
          python hits-preprocess.py [--config CONFIG] [options]

        Options:
          --config PATH              Path to a YAML config file with defaults.
          --input_path PATH          Path to input CSV/XLSX file (required if not set in config).
          --output_path PATH         Directory for processed outputs (default: ./processed_data).
          --to-gist-matrix [bool]    Convert to GIST matrix outputs (default: false).
          --gist_output PATH         Explicit output CSV path for GIST matrix.
          --visualize [bool]         Generate visualizations (default: false).
          --parallel [bool]          Enable multiprocessing pipeline (default: false).
          --scale_activity [bool]    Scale activity values (default: true).
          --convert_units [bool]     Convert measurement units to SI (default: true).
          --correct_pH [bool]        Apply pH correction (default: false).
          --pH_method CHOICE         pH correction method: all|henderson_hasselbalch|empirical|molecular_properties.
          --target_pH FLOAT          Target pH value (default: 7.4).
          --endpoint_mapper CHOICE   Endpoint mapping mode: llm|manual (default: llm).
          --manual_mapping_path PATH Optional CSV/JSON of custom endpoint mappings for manual mode.
          --manual_min_similarity F  Minimum similarity score (0-1) for manual matching (default: 0.55).
          --manual_prefer_exact B    Prefer exact synonym hits before similarity (default: true).
          --split CHOICE             Data split method: random|scaffold|stratified.
          --test_size FLOAT          Test split ratio (default: 0.2).
          --valid_size FLOAT         Validation split ratio (default: 0.1).
          --activity_col NAME        Activity column name (default: Measurement_Value).
          --smiles_col NAME          SMILES column name (default: SMILES_Structure_Parent).
          --debug [bool]             Enable debug logging (default: false).

        Notes:
          • Boolean flags accept true/false values (e.g., --visualize true).
          • CLI arguments always override values loaded from --config.
          • Example: python hits-preprocess.py --config config/preprocess_defaults.yaml --split scaffold
        """
    ).strip("\n")
    print(help_text)


def _safe_merge(*configs: Any) -> DictConfig:
    try:
        return OmegaConf.merge(*configs)
    except (ValidationError, ConfigKeyError) as exc:
        raise ValueError(f"Invalid configuration option: {exc}") from exc


def _ensure_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)

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
        numeric_values = pd.to_numeric(each_df[activity_col].astype(str).str.replace(r'^[<>]=?\s*', '', regex=True), errors='coerce')
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


class UnitConverter:
    """
    Class for converting units to SI units using pint package
    """
    def __init__(self):
        if not PINT_AVAILABLE:
            raise ImportError("pint package is required for unit conversion. Please install it with: pip install pint")
        
        self.ureg = UnitRegistry()
        
        # Common unit mappings for chemical/pharmaceutical data
        self.unit_mappings = {
            # Concentration units
            'ug/ml': 'ug/mL',
            'ug/l': 'ug/L', 
            'mg/ml': 'mg/mL',
            'mg/l': 'mg/L',
            'ng/ml': 'ng/mL',
            'ng/l': 'ng/L',
            'pg/ml': 'pg/mL',
            'pg/l': 'pg/L',
            'um': 'uM',
            'umol/l': 'uM',
            'umol/liter': 'uM',
            'mmol/l': 'mM',
            'mmol/liter': 'mM',
            'nmol/l': 'nM',
            'nmol/liter': 'nM',
            'pmol/l': 'pM',
            'pmol/liter': 'pM',
            'mol/l': 'M',
            'mol/liter': 'M',
            
            # Time units
            'hr': 'hour',
            'hrs': 'hour',
            'hours': 'hour',
            'min': 'minute',
            'mins': 'minute',
            'minutes': 'minute',
            'sec': 'second',
            'secs': 'second',
            'seconds': 'second',
            'day': 'day',
            'days': 'day',
            
            # Volume units
            'ml': 'mL',
            'l': 'L',
            'ul': 'uL',
            'nl': 'nL',
            'pl': 'pL',
            
            # Mass units
            'ug': 'ug',
            'mg': 'mg',
            'ng': 'ng',
            'pg': 'pg',
            'g': 'g',
            'kg': 'kg',
            
            # Percentage
            '%': 'percent',
            'percent': 'percent',
            
            # Temperature
            'c': 'degC',
            'celsius': 'degC',
            'f': 'degF',
            'fahrenheit': 'degF',
            'k': 'kelvin',
            'kelvin': 'kelvin',
            
            # Pressure
            'atm': 'atm',
            'bar': 'bar',
            'pa': 'Pa',
            'pascal': 'Pa',
            'psi': 'psi',
            
            # Energy
            'j': 'J',
            'joule': 'J',
            'kj': 'kJ',
            'kcal': 'kcal',
            'cal': 'cal',
            
            # Length
            'm': 'm',
            'cm': 'cm',
            'mm': 'mm',
            'um': 'um',
            'nm': 'nm',
            'pm': 'pm',
            'angstrom': 'angstrom',
            'a': 'angstrom',
        }
        
        # v4.6 compound units (keyed by lowercased, whitespace-stripped form after
        # '·'->'*' and '^'->'**') mapped to pint-parseable strings.
        self.compound_unit_mappings = {
            'ng*h/ml': 'ng*hour/mL', 'ng*hr/ml': 'ng*hour/mL', 'ng*hour/ml': 'ng*hour/mL',
            'ug*h/ml': 'ug*hour/mL', 'mg*h/ml': 'mg*hour/mL',
            'l/hr/kg': 'L/hour/kg', 'l/h/kg': 'L/hour/kg', 'ml/min/kg': 'mL/minute/kg',
            'ul/min/mg': 'uL/minute/mg', 'ml/min/mg': 'mL/minute/mg',
            'min-1': '1/minute', 'hr-1': '1/hour', 'h-1': '1/hour', 's-1': '1/second',
        }

        # Reporting/dimensionless units whose numeric value must NOT be rescaled
        # (leaving them to pint would e.g. turn 10-6 cm/s into a ~1e8x corruption,
        # or 50% into 0.5). Kept verbatim with an explicit label.
        self.keep_as_is_units = {
            '1e-6 cm/s', 'ratio', 'fold', 'percent', 'dimensionless', 'x',
        }

        # SI unit targets for different measurement types
        self.si_targets = {
            'concentration': 'M',  # Molar (mol/L)
            'time': 's',           # Seconds
            'volume': 'L',         # Liters
            'mass': 'g',           # Grams
            'temperature': 'K',    # Kelvin
            'pressure': 'Pa',      # Pascal
            'energy': 'J',         # Joule
            'length': 'm',         # Meters
            'percentage': 'dimensionless',  # No unit
        }
    
    def normalize_unit_string(self, unit_str):
        """
        Normalize a unit string to a pint-parseable (or keep-as-is) form.

        Unlike the old implementation this PRESERVES '.', '-', '*', '^', '/' and
        '·' so composite/exponent units survive (e.g. '10-6 cm/s', 'ng·h/mL',
        'min-1'). Returns None for blank/placeholder units.
        """
        if unit_str is None or (isinstance(unit_str, float) and pd.isna(unit_str)):
            return None
        u = str(unit_str).strip()
        if u in ('', '-', 'Not specified', 'nan', 'NaN', 'None'):
            return None

        # Molar concentrations are CASE-SENSITIVE: uM/nM/mM/pM/M mean molar, not
        # length (um=micrometre). Resolve them before lowercasing collapses the M.
        molar = re.fullmatch(r'([munp]?)M', u.replace(' ', ''))
        if molar:
            return {'u': 'uM', 'm': 'mM', 'n': 'nM', 'p': 'pM', '': 'M'}[molar.group(1)]

        low = u.lower()
        low = low.replace('·', '*').replace('×', '*').replace('•', '*')
        low = re.sub(r'\s+', '', low)

        # Permeability reporting unit: '10-6 cm/s', '10^-6 cm/s', 'x10-6 cm/s', ...
        if re.fullmatch(r'(x)?10\^?-6cm/s(ec)?', low):
            return '1e-6 cm/s'

        # Dimensionless / reporting units kept verbatim.
        if low in ('ratio', 'fold', 'x'):
            return low
        if low in ('%', 'percent'):
            return 'percent'

        # First-order rate constants: 'min-1' -> '1/minute', etc.
        m = re.fullmatch(r'([a-z]+)-1', low)
        if m and m.group(1) in ('min', 'hr', 'h', 's', 'sec'):
            base = {'min': 'minute', 'hr': 'hour', 'h': 'hour', 's': 'second', 'sec': 'second'}[m.group(1)]
            return f'1/{base}'

        # Units with non-pint tokens (e.g. 'uL/min/10^6 cells'): keep as-is.
        if 'cell' in low:
            return low

        low = low.replace('^', '**')

        # Compound v4.6 units first, then the simple table (both restore casing,
        # e.g. 'um' -> 'uM' so micromolar is not read as micrometre).
        if low in self.compound_unit_mappings:
            return self.compound_unit_mappings[low]
        if low in self.unit_mappings:
            return self.unit_mappings[low]

        return low
    
    def detect_measurement_type(self, unit_str):
        """
        Detect the type of measurement based on unit
        """
        if not unit_str:
            return 'unknown'
            
        unit_str = str(unit_str).lower()

        # Dimensionless / reporting units.
        if unit_str in ('ratio', 'fold', '-', 'x', '%', 'percent') or 'cell' in unit_str:
            return 'dimensionless'

        # Rate / clearance units (per-time, possibly per-mass): check before the
        # concentration/time heuristics so 'mL/min/kg' is not mislabelled.
        if re.search(r'/\s*(min|hr|h|sec|s|day)\b', unit_str) or re.fullmatch(r'1/(min|hour|minute|s|second|hr|h)', unit_str):
            return 'rate'

        # Concentration indicators
        if any(x in unit_str for x in ['/ml', '/l', 'm', 'mol']):
            return 'concentration'

        # Time indicators
        if any(x in unit_str for x in ['hr', 'min', 'sec', 'day']):
            return 'time'
        
        # Volume indicators
        if any(x in unit_str for x in ['ml', 'l', 'ul', 'nl']):
            return 'volume'
        
        # Mass indicators
        if any(x in unit_str for x in ['g', 'kg', 'mg', 'ug', 'ng']):
            return 'mass'
        
        # Temperature indicators
        if any(x in unit_str for x in ['c', 'f', 'k', 'celsius', 'fahrenheit', 'kelvin']):
            return 'temperature'
        
        # Pressure indicators
        if any(x in unit_str for x in ['atm', 'bar', 'pa', 'psi']):
            return 'pressure'
        
        # Energy indicators
        if any(x in unit_str for x in ['j', 'cal', 'kcal']):
            return 'energy'
        
        # Length indicators
        if any(x in unit_str for x in ['m', 'cm', 'mm', 'nm', 'angstrom']):
            return 'length'
        
        # Percentage
        if unit_str in ['%', 'percent']:
            return 'percentage'
        
        return 'unknown'
    
    def convert_to_si(self, value, unit_str):
        """
        Convert a value with unit to SI units
        
        Parameters:
        -----------
        value : float
            The numeric value
        unit_str : str
            The unit string
            
        Returns:
        --------
        tuple : (converted_value, si_unit, original_unit)
        """
        if not PINT_AVAILABLE:
            return value, unit_str, unit_str
            
        try:
            # Normalize unit string
            normalized_unit = self.normalize_unit_string(unit_str)
            if not normalized_unit:
                return value, unit_str, unit_str

            # Reporting/dimensionless units: keep the numeric value unchanged so we
            # never introduce a scale corruption (e.g. 10-6 cm/s, %, ratio, cells).
            if normalized_unit in self.keep_as_is_units or 'cell' in normalized_unit:
                return value, normalized_unit, unit_str

            # Create quantity with pint
            try:
                quantity = value * self.ureg(normalized_unit)
            except Exception:
                logger.warning(
                    f"Could not parse unit '{unit_str}' (normalized '{normalized_unit}'). "
                    f"Keeping original value unconverted.")
                return value, unit_str, unit_str

            # Convert to SI
            si_quantity = quantity.to_base_units()
            si_unit_str = str(si_quantity.units)
            return float(si_quantity.magnitude), si_unit_str, unit_str

        except Exception as e:
            logger.warning(f"Error converting {value} {unit_str} to SI: {e}. Keeping original value.")
            return value, unit_str, unit_str
    
    def convert_column_to_si(self, df, value_col, unit_col, new_value_col=None, new_unit_col=None):
        """
        Convert a column of values with units to SI units
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        value_col : str
            Column name containing numeric values
        unit_col : str
            Column name containing unit strings
        new_value_col : str, optional
            New column name for converted values (default: value_col + '_si')
        new_unit_col : str, optional
            New column name for SI units (default: unit_col + '_si')
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with converted values
        """
        if not PINT_AVAILABLE:
            logger.warning("pint package not available. Returning original data.")
            return df
        
        if new_value_col is None:
            new_value_col = f"{value_col}_si"
        if new_unit_col is None:
            new_unit_col = f"{unit_col}_si"
        
        # Initialize new columns as object dtype so per-row float/str assignment
        # works under pandas 3.0 (value_col is often cast to str upstream, and a
        # str-dtype column rejects float assignment).
        df[new_value_col] = df[value_col].astype(object)
        df[new_unit_col] = df[unit_col].astype(object)
        
        # Track conversion statistics
        converted_count = 0
        total_count = 0
        
        for idx, row in df.iterrows():
            try:
                value = row[value_col]
                unit = row[unit_col]
                
                # Skip if value is not numeric or unit is missing
                if pd.isna(value) or pd.isna(unit) or unit == "Not specified":
                    continue
                
                # Try to convert to numeric if it's a string
                try:
                    if isinstance(value, str):
                        # Remove comparison operators if present (>, >=, <, <=)
                        value = re.sub(r'^[<>]=?\s*', '', value)
                        value = float(value)
                except:
                    continue
                
                total_count += 1
                
                # Convert to SI
                converted_value, si_unit, original_unit = self.convert_to_si(value, unit)
                
                # Update dataframe
                df.at[idx, new_value_col] = converted_value
                df.at[idx, new_unit_col] = si_unit
                
                if si_unit != original_unit:
                    converted_count += 1
                    
            except Exception as e:
                logger.warning(f"Error converting row {idx}: {e}")
                continue
        
        logger.info(f"Unit conversion completed: {converted_count}/{total_count} values converted to SI units")
        return df
    
    def get_conversion_summary(self, df, value_col, unit_col):
        """
        Get a summary of unit conversions in the dataframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        value_col : str
            Column name containing numeric values
        unit_col : str
            Column name containing unit strings
            
        Returns:
        --------
        dict : Summary of unit conversions
        """
        if not PINT_AVAILABLE:
            return {"error": "pint package not available"}
        
        unit_counts = df[unit_col].value_counts()
        measurement_types = {}
        
        for unit, count in unit_counts.items():
            if pd.notna(unit) and unit != "Not specified":
                measurement_type = self.detect_measurement_type(unit)
                if measurement_type not in measurement_types:
                    measurement_types[measurement_type] = {}
                measurement_types[measurement_type][unit] = count
        
        return {
            "total_records": len(df),
            "unit_counts": unit_counts.to_dict(),
            "measurement_types": measurement_types
        }


class pHCorrector:
    """
    Class for pH correction of activity values using different methods
    """
    def __init__(self, method='all', target_pH=7.4):
        """
        Initialize pH corrector
        
        Parameters:
        -----------
        method : str
            Correction method: 'all', 'henderson_hasselbalch', 'empirical', 'molecular_properties'
        target_pH : float
            Target pH for correction (default: 7.4, physiological pH)
        """
        self.method = method
        self.target_pH = target_pH
        
        # Check if RDKit is available for molecular properties method
        try:
            from rdkit.Chem import Descriptors
            self.rdkit_available = True
        except ImportError:
            self.rdkit_available = False
            logger.warning("RDKit not available. Molecular properties method will be disabled.")
    
    def _predict_pKa(self, smiles):
        """
        Predict pKa values for a molecule
        
        Parameters:
        -----------
        smiles : str
            SMILES string of the molecule
            
        Returns:
        --------
        tuple : (pKa_acidic, pKa_basic) or (None, None) if prediction fails
        """
        if not self.rdkit_available:
            return None, None
            
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, None
            
            # Simple pKa estimation based on molecular properties
            # This is a simplified approach - in practice, more sophisticated models would be used
            
            # Count acidic and basic groups
            acidic_groups = 0
            basic_groups = 0
            
            for atom in mol.GetAtoms():
                # Carboxylic acids (typical pKa ~4-5)
                if atom.GetSymbol() == 'O' and atom.GetDegree() == 1:
                    for bond in atom.GetBonds():
                        if bond.GetOtherAtom(atom).GetSymbol() == 'C':
                            acidic_groups += 1
                
                # Amines (typical pKa ~9-10)
                if atom.GetSymbol() == 'N' and atom.GetDegree() <= 3:
                    basic_groups += 1
            
            # Estimate pKa values (simplified)
            pKa_acidic = 4.5 if acidic_groups > 0 else None
            pKa_basic = 9.5 if basic_groups > 0 else None
            
            return pKa_acidic, pKa_basic
            
        except Exception as e:
            logger.warning(f"Error predicting pKa for {smiles}: {e}")
            return None, None
    
    def _henderson_hasselbalch_correction(self, activity_value, pH_measured, pKa_acidic=None, pKa_basic=None):
        """
        Correct activity using Henderson-Hasselbalch equation
        
        Parameters:
        -----------
        activity_value : float
            Measured activity value
        pH_measured : float
            pH at which activity was measured
        pKa_acidic : float, optional
            Acidic pKa of the compound
        pKa_basic : float, optional
            Basic pKa of the compound
            
        Returns:
        --------
        float : Corrected activity value
        """
        if pH_measured == self.target_pH:
            return activity_value
        
        # Determine if compound is primarily acidic or basic
        is_acidic = False
        pKa = None
        
        if pKa_acidic is not None and pKa_basic is not None:
            # Use the pKa closer to physiological pH
            if abs(pKa_acidic - 7.4) < abs(pKa_basic - 7.4):
                pKa = pKa_acidic
                is_acidic = True
            else:
                pKa = pKa_basic
                is_acidic = False
        elif pKa_acidic is not None:
            pKa = pKa_acidic
            is_acidic = True
        elif pKa_basic is not None:
            pKa = pKa_basic
            is_acidic = False
        else:
            # No pKa available, return original value
            return activity_value
        
        try:
            # Calculate ionization fractions
            if is_acidic:
                # Acidic drug: HA ↔ H+ + A-
                f_ionized_measured = 1 / (1 + 10**(pKa - pH_measured))
                f_ionized_target = 1 / (1 + 10**(pKa - self.target_pH))
            else:
                # Basic drug: B + H+ ↔ BH+
                f_ionized_measured = 1 / (1 + 10**(pH_measured - pKa))
                f_ionized_target = 1 / (1 + 10**(self.target_pH - pKa))
            
            # Unionized fractions
            f_unionized_measured = 1 - f_ionized_measured
            f_unionized_target = 1 - f_ionized_target
            
            # Avoid division by zero
            if f_unionized_measured == 0:
                return activity_value
            
            # Correction factor
            correction_factor = f_unionized_target / f_unionized_measured
            
            # Corrected activity value
            corrected_activity = activity_value * correction_factor
            
            return corrected_activity
            
        except Exception as e:
            logger.warning(f"Error in Henderson-Hasselbalch correction: {e}")
            return activity_value
    
    def _empirical_correction(self, activity_value, pH_measured):
        """
        Correct activity using empirical pH-activity relationship
        
        Parameters:
        -----------
        activity_value : float
            Measured activity value
        pH_measured : float
            pH at which activity was measured
            
        Returns:
        --------
        float : Corrected activity value
        """
        if pH_measured == self.target_pH:
            return activity_value
        
        try:
            # Simple empirical correction based on pH difference
            # This is a simplified approach - in practice, more sophisticated models would be used
            
            pH_diff = self.target_pH - pH_measured
            
            # Empirical correction factor based on pH difference
            # Values are estimated based on typical pH effects on drug activity
            if abs(pH_diff) <= 1.0:
                # Small pH difference: minimal correction
                correction_factor = 1.0 + (pH_diff * 0.05)
            elif abs(pH_diff) <= 2.0:
                # Medium pH difference: moderate correction
                correction_factor = 1.0 + (pH_diff * 0.1)
            else:
                # Large pH difference: significant correction
                correction_factor = 1.0 + (pH_diff * 0.15)
            
            # Limit correction factor to reasonable range
            correction_factor = max(0.1, min(10.0, correction_factor))
            
            corrected_activity = activity_value * correction_factor
            
            return corrected_activity
            
        except Exception as e:
            logger.warning(f"Error in empirical correction: {e}")
            return activity_value
    
    def _molecular_properties_correction(self, activity_value, pH_measured, smiles):
        """
        Correct activity using molecular properties
        
        Parameters:
        -----------
        activity_value : float
            Measured activity value
        pH_measured : float
            pH at which activity was measured
        smiles : str
            SMILES string of the molecule
            
        Returns:
        --------
        float : Corrected activity value
        """
        if not self.rdkit_available or pH_measured == self.target_pH:
            return activity_value
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return activity_value
            
            # Calculate molecular properties
            mol_weight = Descriptors.ExactMolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Calculate pH sensitivity score based on molecular properties
            # Higher logP: more lipophilic, less pH sensitive
            # Higher TPSA: more polar, more pH sensitive
            # More HBD/HBA: more pH sensitive
            
            logp_factor = max(0.5, min(2.0, 1.0 - (logp / 10.0)))  # LogP effect
            tpsa_factor = max(0.5, min(2.0, 1.0 + (tpsa / 200.0)))  # TPSA effect
            hbd_hba_factor = max(0.5, min(2.0, 1.0 + ((hbd + hba) / 20.0)))  # HBD/HBA effect
            
            # Combined pH sensitivity
            pH_sensitivity = (logp_factor + tpsa_factor + hbd_hba_factor) / 3.0
            
            # Calculate correction based on pH difference and sensitivity
            pH_diff = self.target_pH - pH_measured
            correction_factor = 1.0 + (pH_diff * 0.1 * pH_sensitivity)
            
            # Limit correction factor
            correction_factor = max(0.1, min(10.0, correction_factor))
            
            corrected_activity = activity_value * correction_factor
            
            return corrected_activity
            
        except Exception as e:
            logger.warning(f"Error in molecular properties correction: {e}")
            return activity_value
    
    def correct_activity(self, activity_value, pH_measured, smiles=None):
        """
        Correct activity value using the specified method
        
        Parameters:
        -----------
        activity_value : float
            Measured activity value
        pH_measured : float
            pH at which activity was measured
        smiles : str, optional
            SMILES string of the molecule (required for molecular properties method)
            
        Returns:
        --------
        dict : Dictionary with corrected values for each method
        """
        if pd.isna(activity_value) or pd.isna(pH_measured):
            return {
                'original': activity_value,
                'henderson_hasselbalch': activity_value,
                'empirical': activity_value,
                'molecular_properties': activity_value
            }
        
        try:
            activity_value = float(activity_value)
            pH_measured = float(pH_measured)
        except (ValueError, TypeError):
            return {
                'original': activity_value,
                'henderson_hasselbalch': activity_value,
                'empirical': activity_value,
                'molecular_properties': activity_value
            }
        
        result = {'original': activity_value}
        
        # Henderson-Hasselbalch correction
        if self.method in ['all', 'henderson_hasselbalch']:
            pKa_acidic, pKa_basic = self._predict_pKa(smiles) if smiles else (None, None)
            result['henderson_hasselbalch'] = self._henderson_hasselbalch_correction(
                activity_value, pH_measured, pKa_acidic, pKa_basic
            )
        
        # Empirical correction
        if self.method in ['all', 'empirical']:
            result['empirical'] = self._empirical_correction(activity_value, pH_measured)
        
        # Molecular properties correction
        if self.method in ['all', 'molecular_properties']:
            result['molecular_properties'] = self._molecular_properties_correction(
                activity_value, pH_measured, smiles
            )
        
        return result
    
    def correct_column(self, df, activity_col, pH_col, smiles_col=None):
        """
        Correct activity values in a DataFrame column
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        activity_col : str
            Column name containing activity values
        pH_col : str
            Column name containing pH values
        smiles_col : str, optional
            Column name containing SMILES strings
            
        Returns:
        --------
        pandas.DataFrame : DataFrame with corrected values
        """
        if activity_col not in df.columns or pH_col not in df.columns:
            logger.warning(f"Required columns not found: {activity_col}, {pH_col}")
            return df
        
        # Create new columns for corrected values
        if self.method == 'all':
            new_cols = [
                f"{activity_col}_pH_corrected_hh",
                f"{activity_col}_pH_corrected_emp",
                f"{activity_col}_pH_corrected_mp"
            ]
        else:
            method_suffix = {
                'henderson_hasselbalch': 'hh',
                'empirical': 'emp',
                'molecular_properties': 'mp'
            }
            new_cols = [f"{activity_col}_pH_corrected_{method_suffix[self.method]}"]
        
        # Initialize new columns as object dtype so per-row float assignment works
        # under pandas 3.0 (activity_col is often str-typed upstream).
        for col in new_cols:
            df[col] = df[activity_col].astype(object)
        
        corrected_count = 0
        total_count = 0
        
        for idx, row in df.iterrows():
            try:
                activity = row[activity_col]
                pH = row[pH_col]
                smiles = row[smiles_col] if smiles_col else None
                
                if pd.isna(activity) or pd.isna(pH):
                    continue
                
                total_count += 1
                
                # Perform correction
                corrected = self.correct_activity(activity, pH, smiles)
                
                # Update dataframe
                if self.method == 'all':
                    df.at[idx, f"{activity_col}_pH_corrected_hh"] = corrected['henderson_hasselbalch']
                    df.at[idx, f"{activity_col}_pH_corrected_emp"] = corrected['empirical']
                    df.at[idx, f"{activity_col}_pH_corrected_mp"] = corrected['molecular_properties']
                else:
                    method_key = self.method
                    df.at[idx, new_cols[0]] = corrected[method_key]
                
                if corrected['original'] != corrected.get('henderson_hasselbalch', corrected['original']):
                    corrected_count += 1
                    
            except Exception as e:
                logger.warning(f"Error correcting row {idx}: {e}")
                continue
        
        logger.info(f"pH correction completed: {corrected_count}/{total_count} values corrected using {self.method} method")
        return df


def process_smiles_batch_global(smiles_batch):
    """
    Global function for processing SMILES batches in parallel.
    This function is defined at module level to avoid pickle issues with multiprocessing.
    """
    from rdkit import Chem
    from rdkit.Chem import MolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
    
    # Initialize RDKit objects locally to avoid pickle issues
    # Use the same approach as the main class
    uncharger = MolStandardize.rdMolStandardize.Uncharger()
    enumerator = MolStandardize.rdMolStandardize.TautomerEnumerator()
    
    results = []
    for smiles in smiles_batch:
        try:
            # Standardize SMILES using the same approach as preprocess_compound
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"SMILES string is not valid: {smiles}")

            # Remove isotope
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)

            mol = uncharger.uncharge(mol)                  # Uncharge
            Chem.SetAromaticity(mol)                        # Aromaticity
            mol = Chem.RemoveHs(mol)                        # Remove H

            # Stereochemistry (keep stereo by default)
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
            
            mol = enumerator.Canonicalize(mol)             # Tautomer
            
            try:
                Chem.SanitizeMol(mol)                       # Check valence error
            except Chem.rdChem.KekulizeException:
                raise ValueError(f"Valence error detected in molecule: {smiles}")
            
            standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)    # Scaffold extraction
            scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
            
            results.append((standardized_smiles, scaffold_smiles))
        except Exception as e:
            logger.warning(f"Error processing SMILES {smiles}: {e}")
            results.append((None, None))
    return results


def _parse_bracket_numbers(cell):
    """Parse a Permeability cell into a list of floats/None.

    Accepts bracketed lists like '[6,2]', '[6,]', '[,2]', plain scalars '6.2',
    or empty/placeholder values. An empty slot (e.g. the '' in '[6,]') becomes
    None so directional presence is preserved.
    """
    if cell is None:
        return []
    s = str(cell).strip()
    if s in ("", "-", "nan", "NaN", "None", "Not specified"):
        return []
    s = s.strip("[]")
    if s == "":
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            out.append(None)
            continue
        part = re.sub(r'^[<>]=?\s*', '', part)  # tolerate leading comparators
        try:
            out.append(float(part))
        except ValueError:
            out.append(None)
    return out


def parse_permeability_pair(atob_cell, btoa_cell):
    """Return (AtoB, BtoA) floats from the two Permeability value columns.

    Handles the v4.6 '[Value, Value]=[AB, BA]' convention whether the pair is
    split across the AtoB/BtoA columns or carried as a full list in either one.
    Missing directions come back as None.
    """
    ab = _parse_bracket_numbers(atob_cell)
    ba = _parse_bracket_numbers(btoa_cell)
    atob = ab[0] if len(ab) >= 1 else None
    btoa = ba[1] if len(ba) >= 2 else (ba[0] if len(ba) == 1 else None)
    # Fall back to the other column if one carries the full [AB, BA] pair.
    if atob is None and len(ba) >= 1:
        atob = ba[0]
    if btoa is None and len(ab) >= 2:
        btoa = ab[1]
    return atob, btoa


class DataInspector:
    def __init__(self,
                 input_path:str,
                 smiles_column:str='SMILES_Structure_Parent',
                 activity_column:str='Measurement_Value',
                 condition_columns:list=['Test', 'Test_Type', 'Test_Subject', 'Measurement_Type', 'Measurement_Conc', 'Measurement_Temp', 'Measurement_Class', 'Measurement_Route', 'Measurement_Sex', 'Measurement_Formulation']
                 ):
        self.smiles_col = smiles_column
        self.activity_col = activity_column
        self.condition_columns = condition_columns
        self.is_human_pk = False
        self.df = self.load_data(input_path)
    
    # Canonical column names keyed by their normalized form (lowercased, trailing
    # '*' and surrounding whitespace stripped). Covers the v4.6 spelling, the
    # legacy 'Measurment_*' typo, the 'Test_Species' -> 'Test_Subject' rename, and
    # VIVO's lowercase 'Measurement_value'. Case-insensitivity is what fixes B1/B4/H5.
    CANONICAL_COLUMNS = {
        'chemical id': 'Chemical ID',
        'chemical name': 'Chemical Name',
        'smiles_structure_parent': 'SMILES_Structure_Parent',
        'smiles_salt': 'SMILES_Salt',
        'test': 'Test',
        'test_type': 'Test_Type',
        'test_subject': 'Test_Subject',
        'test_species': 'Test_Subject',          # legacy rename
        'test_dose': 'Test_Dose',
        'measurement_type': 'Measurement_Type',
        'measurment_type': 'Measurement_Type',   # legacy typo
        'measurement_relation': 'Measurement_Relation',
        'measurment_relation': 'Measurement_Relation',
        'measurement_value': 'Measurement_Value',
        'measurment_value': 'Measurement_Value',
        'measurement_value(atob)': 'Measurement_Value(AtoB)',
        'measurement_value(btoa)': 'Measurement_Value(BtoA)',
        'measurement_unit': 'Measurement_Unit',
        'measurment_unit': 'Measurement_Unit',
        'measurement_conc': 'Measurement_Conc',
        'measurement_temp': 'Measurement_Temp',
        'measurement_condition': 'Measurement_Condition',
        'measurment_condition': 'Measurement_Condition',
        'measurement_class': 'Measurement_Class',
        'measurement_route': 'Measurement_Route',
        'measurement_sex': 'Measurement_Sex',
        'measurement_formulation': 'Measurement_Formulation',
        'comment': 'Comment',
    }

    def normalize_column_names(self, df):
        """
        Normalize column names to canonical K-MELLODDY names.

        Handles v4.6 (`Measurement_*`), the legacy typo (`Measurment_*`), the
        `Test_Subject*`/`Test_Species` variants and VIVO's lowercase
        `Measurement_value` via case-insensitive canonicalization. The real value
        column is preserved; the legacy `Unnamed: 10` remap is applied only as a
        fallback when no `Measurement_Value` is present (old merged-cell layout).
        """
        rename_map = {}
        for col in df.columns:
            key = str(col).strip().rstrip('*').strip().lower()
            canonical = self.CANONICAL_COLUMNS.get(key)
            if canonical and canonical != col:
                rename_map[col] = canonical
        if rename_map:
            df = df.rename(columns=rename_map)

        # Drop duplicate columns produced by renaming (keep the first occurrence).
        df = df.loc[:, ~df.columns.duplicated()]

        # Legacy fallback: some old Excel exports carried the real numbers in an
        # 'Unnamed: 10' column next to a placeholder value column. Only use it when
        # a real Measurement_Value did not survive canonicalization.
        if 'Measurement_Value' not in df.columns and 'Unnamed: 10' in df.columns:
            df = df.rename(columns={'Unnamed: 10': 'Measurement_Value'})

        # Remove any remaining unnamed columns.
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        logger.info(f"Normalized column names. Final columns: {df.columns.tolist()}")
        return df
    
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

            # Human PK: wide-format, 3-row header (real header at row index 2),
            # separate table sheets, and a non-GIST schema. Load the raw tables
            # without ADMET-style normalization; the caller routes it to a
            # separate output.
            human_pk_sheets = [s for s in sheets
                               if str(s).startswith('데이터') and 'human pk' in str(s).lower()]
            if human_pk_sheets:
                self.is_human_pk = True
                logger.warning(
                    "Human PK workbook detected (sheets: %s). Human PK is a "
                    "wide-format, non-GIST schema; loading raw table without "
                    "endpoint mapping.", ", ".join(human_pk_sheets))
                hpk_dfs = [pd.read_excel(excel_file, sheet_name=s, header=2).fillna("Not specified")
                           for s in human_pk_sheets]
                df = pd.concat(hpk_dfs, ignore_index=True)
                df = df.loc[:, ~df.columns.duplicated()]
                df = self._map_human_pk_smiles(df)
                for col in df.columns:
                    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col].dtype):
                        df[col] = df[col].astype(str).str.replace('μ', 'u')
                        df[col] = df[col].apply(lambda x: self._replace_special_chars(x) if isinstance(x, str) else x)
                return df

            dfs = []
            if 'ADMET' in sheets:
                logger.info("Reading ADMET sheet")
                admet_df = pd.read_excel(excel_file, sheet_name='ADMET').fillna("Not specified")
                dfs.append(admet_df)
            
            if 'PK' in sheets:
                logger.info("Reading PK sheet")
                pk_df = pd.read_excel(excel_file, sheet_name='PK').fillna("Not specified")
                dfs.append(pk_df)
            
            # Check for new K-MELLODDY format with 'Data' sheet
            if '데이터' in sheets:
                logger.info("Reading Data sheet (new K-MELLODDY format)")
                data_df = pd.read_excel(excel_file, sheet_name='데이터', header=1).fillna("Not specified")
                dfs.append(data_df)
                
            if not dfs:
                # If no specific sheets found, read the first sheet
                logger.warning(f"No ADMET, PK, or Data sheets found in {input_path}. Reading first sheet.")
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
            if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col].dtype):  # Only process string columns
                # Replace μ (micro) with u
                df[col] = df[col].astype(str).str.replace('μ', 'u')
                # Replace other potentially problematic unicode characters
                df[col] = df[col].apply(lambda x: self._replace_special_chars(x) if isinstance(x, str) else x)
                
        # Normalize column names to handle both old and new formats
        df = self.normalize_column_names(df)

        # Permeability: split AtoB/BtoA list values into a usable value column.
        df = self._expand_permeability_columns(df)

        # Ensure all condition columns exist
        for col in self.condition_columns:
            if col not in df.columns:
                logger.warning(f"Condition column '{col}' not found in data. Creating empty column.")
                df[col] = "Not specified"

        return df

    def _map_human_pk_smiles(self, df):
        """Rename the Human PK SMILES column (e.g. 'Substance\\n (약물, SMILES)')
        to the canonical SMILES column so schema validation passes."""
        if self.smiles_col in df.columns:
            return df
        for col in df.columns:
            if 'smiles' in str(col).lower():
                df = df.rename(columns={col: self.smiles_col})
                logger.info(f"Human PK: mapped SMILES column '{col}' -> '{self.smiles_col}'.")
                break
        return df

    def _expand_permeability_columns(self, df):
        """Turn Permeability's ``Measurement_Value(AtoB)``/``(BtoA)`` list columns
        into a scalar ``Measurement_Value`` (AtoB = the primary permeability value)
        plus a ``Measurement_EffluxRatio`` helper (BtoA/AtoB) for GIST derivation.
        """
        ab_col, ba_col = 'Measurement_Value(AtoB)', 'Measurement_Value(BtoA)'
        if ab_col not in df.columns and ba_col not in df.columns:
            return df
        atob_vals, btoa_vals, ratio_vals = [], [], []
        for _, row in df.iterrows():
            atob, btoa = parse_permeability_pair(row.get(ab_col), row.get(ba_col))
            atob_vals.append(atob)
            btoa_vals.append(btoa)
            if atob not in (None, 0) and btoa is not None:
                ratio_vals.append(btoa / atob)
            else:
                ratio_vals.append(None)
        if 'Measurement_Value' not in df.columns:
            df['Measurement_Value'] = atob_vals
        else:
            existing = df['Measurement_Value']
            df['Measurement_Value'] = [
                a if (pd.isna(e) or str(e) in ("", "Not specified")) else e
                for e, a in zip(existing, atob_vals)
            ]
        df['Measurement_Value_BtoA'] = btoa_vals
        df['Measurement_EffluxRatio'] = ratio_vals
        n = sum(1 for v in atob_vals if v is not None)
        logger.info(f"Permeability columns expanded: {n} AtoB values, "
                    f"{sum(1 for v in ratio_vals if v is not None)} efflux ratios derived.")
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
                # Censored = {>, >=, <, <=}. Equality is stored as the quoted
                # literal '"="' in v4.6, so strip quotes before comparing; '=' and
                # blanks count as uncensored.
                rel_norm = (each_df['Measurement_Relation'].astype(str)
                            .str.strip().str.strip('"').str.strip("'").str.strip())
                censored_mask = rel_norm.isin(['>', '>=', '<', '<='])
                uncensored_count = int((~censored_mask).sum())
            else:
                has_relation_mask = each_df[self.activity_col].astype(str).str.contains(r'^[<>]=?')
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
                 convert_units:bool=True,
                 correct_pH:bool=True,
                 pH_method:str='all',
                 target_pH:float=7.4,
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
        self.convert_units = convert_units
        self.correct_pH = correct_pH
        self.pH_method = pH_method
        self.target_pH = target_pH
        self.active_is_high:bool = True
        self.final_cols = []
        self.smiles_column = smiles_column
        self.remover = SaltRemover.SaltRemover()
        self.uncharger = MolStandardize.rdMolStandardize.Uncharger()
        self.enumerator = MolStandardize.rdMolStandardize.TautomerEnumerator()
        
        # Initialize unit converter if needed
        if self.convert_units and PINT_AVAILABLE:
            try:
                self.unit_converter = UnitConverter()
                logger.info("Unit converter initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize unit converter: {e}")
                self.convert_units = False
        elif self.convert_units and not PINT_AVAILABLE:
            logger.warning("pint package not available. Unit conversion disabled.")
            self.convert_units = False
        
        # Initialize pH corrector if needed
        if self.correct_pH:
            try:
                self.pH_corrector = pHCorrector(method=self.pH_method, target_pH=self.target_pH)
                logger.info(f"pH corrector initialized successfully with method: {self.pH_method}")
            except Exception as e:
                logger.warning(f"Failed to initialize pH corrector: {e}")
                self.correct_pH = False
        
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
            
            batch_size = max(100, len(smiles_list) // (multiprocessing.cpu_count() * 2))
            smiles_batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]

            start_time = time.time()
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.map(process_smiles_batch_global, smiles_batches)

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
                
            # Apply scaling (numeric only). Use the pandas dtype API: under
            # pandas 3.0 np.issubdtype(StringDtype, np.number) raises.
            is_numeric = pd.api.types.is_numeric_dtype(labels)
            has_numeric_strings = False
            if not is_numeric:
                try:
                    has_numeric_strings = labels.astype(str).str.match(r'^[<>]?=?\s*\d+\.?\d*$').any()
                except Exception:
                    has_numeric_strings = False
            if is_numeric or has_numeric_strings:
                self.scale_experiment_values(labels)
            
            # Convert continuous values -> classification (using threshold)
            if self.label_type == 'Continuous' and self.task_type == "classification" and self.threshold is not None:
                self.create_classification_label(labels)
                
        except Exception as e:
            logger.error(f"Error preprocessing labels: {e}")
            # Add the base column to the final column list
            if self.activity_col not in self.final_cols:
                self.final_cols.append(self.activity_col)

    def _infer_ph_column(self):
        """Infer a per-row pH value from v4.6 locations, in priority order:
        dedicated pH columns, Test_Subject ('pH7.4'), Measurement_Condition
        (bare number or 'pH x'), then Test_Type. Returns a float Series (NaN where
        no pH is present)."""
        ph_pattern = re.compile(r"pH\s*([0-9]+(?:\.[0-9]+)?)", flags=re.IGNORECASE)

        def _ph_from_text(text):
            m = ph_pattern.search(str(text))
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return None
            return None

        def _bare_ph(text):
            m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*", str(text))
            if m:
                try:
                    v = float(m.group(1))
                    if 0.0 <= v <= 14.0:
                        return v
                except ValueError:
                    return None
            return None

        inferred = pd.Series(np.nan, index=self.df.index, dtype=float)
        for col in ['pH', 'pH_Value', 'Measurement_pH', 'Test_pH']:
            if col in self.df.columns:
                inferred = inferred.combine_first(pd.to_numeric(self.df[col], errors='coerce'))
        if 'Test_Subject' in self.df.columns:
            inferred = inferred.combine_first(
                pd.to_numeric(self.df['Test_Subject'].apply(_ph_from_text), errors='coerce'))
        if 'Measurement_Condition' in self.df.columns:
            mc = self.df['Measurement_Condition'].apply(
                lambda x: _ph_from_text(x) if _ph_from_text(x) is not None else _bare_ph(x))
            inferred = inferred.combine_first(pd.to_numeric(mc, errors='coerce'))
        if 'Test_Type' in self.df.columns:
            inferred = inferred.combine_first(
                pd.to_numeric(self.df['Test_Type'].apply(_ph_from_text), errors='coerce'))
        return inferred

    def preprocess(self):
        compounds = self.df[self.smiles_col]
        labels = self.df[self.activity_col]
        
        # Perform unit conversion if enabled and unit column exists
        if self.convert_units and 'Measurement_Unit' in self.df.columns:
            logger.info("Starting unit conversion to SI units...")
            try:
                # Get conversion summary before conversion
                summary = self.unit_converter.get_conversion_summary(self.df, self.activity_col, 'Measurement_Unit')
                logger.info(f"Unit conversion summary: {summary}")
                
                # Perform unit conversion
                self.df = self.unit_converter.convert_column_to_si(
                    self.df, 
                    self.activity_col, 
                    'Measurement_Unit',
                    f"{self.activity_col}_si",
                    'Measurement_Unit_si'
                )
                
                # Add converted columns to final columns
                if f"{self.activity_col}_si" in self.df.columns:
                    self.final_cols.append(f"{self.activity_col}_si")
                if 'Measurement_Unit_si' in self.df.columns:
                    self.final_cols.append('Measurement_Unit_si')
                
                logger.info("Unit conversion completed successfully")
                
            except Exception as e:
                logger.error(f"Error during unit conversion: {e}")
                logger.info("Continuing without unit conversion")
        
        # Perform pH correction if enabled and pH-related data exists
        if self.correct_pH:
            # Build a unified pH source column from several v4.6 locations in
            # priority order: dedicated pH columns -> Test_Subject ('pH7.4') ->
            # Measurement_Condition (bare number or 'pH x') -> Test_Type embedded.
            inferred_pH = self._infer_ph_column()
            pH_data_mask = inferred_pH.notna()

            if pH_data_mask.any():
                logger.info(f"pH source detected for {int(pH_data_mask.sum())} rows. Starting pH correction...")
                try:
                    self.df['inferred_pH'] = inferred_pH
                    pH_col = 'inferred_pH'
                    if pH_col is not None:
                        logger.info(f"Using pH column: {pH_col}")
                        
                        # Perform pH correction only on pH-related data
                        pH_df = self.df[pH_data_mask].copy()
                        corrected_pH_df = self.pH_corrector.correct_column(
                            pH_df, 
                            self.activity_col, 
                            pH_col, 
                            self.smiles_col
                        )
                        
                        # Update the original dataframe with corrected values
                        for idx in corrected_pH_df.index:
                            for col in corrected_pH_df.columns:
                                if col.startswith(f"{self.activity_col}_pH_corrected"):
                                    self.df.at[idx, col] = corrected_pH_df.at[idx, col]
                        
                        # Add pH corrected columns to final columns
                        if self.pH_method == 'all':
                            pH_corrected_cols = [
                                f"{self.activity_col}_pH_corrected_hh",
                                f"{self.activity_col}_pH_corrected_emp",
                                f"{self.activity_col}_pH_corrected_mp"
                            ]
                        else:
                            method_suffix = {
                                'henderson_hasselbalch': 'hh',
                                'empirical': 'emp',
                                'molecular_properties': 'mp'
                            }
                            pH_corrected_cols = [f"{self.activity_col}_pH_corrected_{method_suffix[self.pH_method]}"]
                        
                        for col in pH_corrected_cols:
                            if col in self.df.columns:
                                self.final_cols.append(col)
                        
                        logger.info(f"pH correction completed successfully for {pH_data_mask.sum()} records")
                        
                except Exception as e:
                    logger.error(f"Error during pH correction: {e}")
                    logger.info("Continuing without pH correction")
            else:
                logger.info("No pH-related data found. pH correction skipped.")
        
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

            # Keep a copy so we can recover WITHOUT losing enriched columns
            # (_si, pH_corrected, Chemical ID) if the strict dedup empties the set.
            df_before_dedup = self.df.copy()

            # Remove duplicates (remove all duplicates with keep=False)
            self.df.drop_duplicates(subset=["Standardized_SMILES"], keep=False, inplace=True, ignore_index=True)

            # Record data count after duplicate removal
            after_count = len(self.df)
            if after_count == 0 and before_count > 0:
                logger.warning(f"All data removed during duplicate removal! Original count: {before_count}")
                # Recover from the pre-dedup frame with keep='first' so all columns
                # (including pH/SI-derived ones) survive.
                self.df = df_before_dedup.drop_duplicates(
                    subset=["Standardized_SMILES"], keep='first', ignore_index=True)
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

        # Select only columns that survived processing (dedup recovery paths can
        # rebuild self.df and drop some expected columns); warn rather than crash.
        existing_cols = [c for c in dict.fromkeys(self.final_cols) if c in self.df.columns]
        missing_cols = [c for c in dict.fromkeys(self.final_cols) if c not in self.df.columns]
        if missing_cols:
            logger.warning(f"Output is missing {len(missing_cols)} expected column(s): {missing_cols}")
        return self.df[existing_cols]

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
            
        # Filter valid SMILES - handle duplicates
        valid_smiles_df = self.df[self.df[self.smiles_col].notna()].copy()
        if len(valid_smiles_df) == 0:
            logger.warning("No valid SMILES found for visualization")
            return None
            
        # Handle duplicate SMILES if present
        if len(valid_smiles_df) > len(valid_smiles_df[self.smiles_col].unique()):
            logger.info(f"Found {len(valid_smiles_df) - len(valid_smiles_df[self.smiles_col].unique())} duplicate SMILES. Keeping first occurrence of each.")
            # Keep only the first row for each unique SMILES
            valid_smiles_df = valid_smiles_df.drop_duplicates(subset=[self.smiles_col], keep='first')
        
        # Limit number of molecules to display
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


std2gist = {
    ""
}

if __name__ == "__main__":
    raw_args = sys.argv[1:]
    if any(flag in raw_args for flag in ("-h", "--help")):
        _print_help()
        sys.exit(0)

    default_cfg: DictConfig = OmegaConf.structured(PreprocessConfig)
    cli_overrides = _convert_argparse_style_to_cli(raw_args)
    try:
        cli_conf = OmegaConf.from_cli(cli_overrides)
    except Exception as exc:
        raise ValueError(f"Failed to parse command line arguments: {exc}") from exc

    # Merge defaults with CLI to discover config path
    merged_cfg: DictConfig = _safe_merge(default_cfg, cli_conf)
    config_path = merged_cfg.get("config")

    if config_path:
        config_path = os.path.expanduser(config_path)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        file_cfg = OmegaConf.load(config_path)
        args: DictConfig = _safe_merge(default_cfg, file_cfg, cli_conf)
    else:
        args = merged_cfg

    if OmegaConf.is_missing(args, "input_path") or not args.input_path:
        raise ValueError("`input_path` must be provided via CLI or configuration file.")

    allowed_pH_methods = {"all", "henderson_hasselbalch", "empirical", "molecular_properties"}
    if args.pH_method not in allowed_pH_methods:
        raise ValueError(f"Invalid pH_method '{args.pH_method}'. Choose from: {sorted(allowed_pH_methods)}")

    if args.split is not None:
        allowed_splits = {"random", "scaffold", "stratified"}
        if args.split not in allowed_splits:
            raise ValueError(f"Invalid split '{args.split}'. Choose from: {sorted(allowed_splits)}")
    
    allowed_mappers = {"llm", "manual"}
    if args.endpoint_mapper not in allowed_mappers:
        raise ValueError(f"Invalid endpoint_mapper '{args.endpoint_mapper}'. Choose from: {sorted(allowed_mappers)}")
    
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

    # Early path: GIST matrix export
    if args.to_gist_matrix:
        try:
            # Add converters to path and import
            llm_src = os.path.join(os.path.dirname(__file__), "llm_converter", "src")
            if llm_src not in sys.path:
                sys.path.append(llm_src)
            from manual_converter import ManualConversionConfig, ManualFormatConverter
            try:
                from format_converter import ConversionConfig, LLMFormatConverter  # type: ignore
            except ImportError:
                ConversionConfig = None  # type: ignore
                LLMFormatConverter = None  # type: ignore
                logger.warning("format_converter module not available; LLM mapping disabled.")
            
            # Load data using existing inspector (handles Measurement_Value actual column mapping)
            inspector = DataInspector(
                input_path=args.input_path,
                smiles_column=args.smiles_col,
                activity_column=args.activity_col
            )
            df = inspector.df.copy()

            # Human PK is a wide-format, non-GIST schema: write it out separately
            # instead of forcing it through endpoint mapping (which would yield an
            # empty/garbage matrix).
            if getattr(inspector, "is_human_pk", False):
                base = os.path.splitext(os.path.basename(args.input_path))[0]
                out_dir = "./processed_data"
                os.makedirs(out_dir, exist_ok=True)
                hpk_path = os.path.join(out_dir, f"{base}_HumanPK.csv")
                df.to_csv(hpk_path, index=False)
                logger.warning(
                    "Human PK detected: wrote raw wide-format table to %s and "
                    "skipped GIST matrix conversion (not a GIST-mappable schema).",
                    hpk_path)
                logger.info("Finished GIST matrix conversion")
                sys.exit(0)

            # Normalize critical column names presence
            smiles_col = args.smiles_col if args.smiles_col in df.columns else "SMILES_Structure_Parent"
            unit_col = "Measurement_Unit" if "Measurement_Unit" in df.columns else None

            # Build canonical endpoint string using Test + Test_Type + Measurement_Type when available
            def _safe_str(x):
                return str(x) if (x is not None and x != "Not specified") else ""
            c1 = df["Test"].apply(_safe_str) if "Test" in df.columns else ""
            c2 = df["Test_Type"].apply(_safe_str) if "Test_Type" in df.columns else ""
            c3 = df["Measurement_Type"].apply(_safe_str) if "Measurement_Type" in df.columns else (df["Measurment_Type"].apply(_safe_str) if "Measurment_Type" in df.columns else "")
            df["endpoint_canonical"] = (c1.astype(str) + " | " + c2.astype(str) + " | " + c3.astype(str)).str.strip(" |")
            df.loc[df["endpoint_canonical"] == "", "endpoint_canonical"] = df.get("Test", "")

            manual_config = ManualConversionConfig(
                mapping_path=args.manual_mapping_path,
                min_similarity=float(args.manual_min_similarity),
                prefer_exact=_ensure_bool(args.manual_prefer_exact),
            )
            manual_converter = ManualFormatConverter(manual_config)

            mapper_choice = args.endpoint_mapper.lower()
            mapping_strategy = mapper_choice
            converter = None

            if mapper_choice == "llm" and LLMFormatConverter and ConversionConfig:
                config = ConversionConfig(
                    api_key=os.getenv("GEMINI_API_KEY", ""),
                    model_name=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
                    temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
                    source=os.getenv("LLM_SOURCE", "Gemini")
                )
                if not config.api_key:
                    logger.warning("GEMINI_API_KEY environment variable is not set. Falling back to manual endpoint mapping.")
                else:
                    try:
                        converter = LLMFormatConverter(config)
                        mapping_strategy = "llm"
                    except Exception as converter_error:
                        logger.warning("LLM converter initialization failed (%s). Using manual mapping.", converter_error)
                        converter = None

            if converter is None:
                mapping_strategy = "manual"
                converter = manual_converter

            tmp_df = pd.DataFrame({"endpoint": df["endpoint_canonical"].unique().tolist()})
            endpoint_map = converter.match_endpoints(tmp_df)
            df["gist_endpoint"] = df["endpoint_canonical"].map(endpoint_map).fillna(df["endpoint_canonical"])  # fallback
            logger.info(f"Endpoint mapping strategy used: {mapping_strategy.upper()}")

            # Unit conversion to SI
            if unit_col is not None:
                uc = UnitConverter()
                df = uc.convert_column_to_si(df, value_col=args.activity_col, unit_col=unit_col, new_value_col="value_si", new_unit_col="unit_si")
                value_col = "value_si"
            else:
                value_col = args.activity_col

            # H6: qualitative results (Ames/Genetoxicity) store Positive/Negative
            # in Measurement_Unit with a '-' value. Encode as 1.0/0.0 before the
            # numeric coercion so they are not dropped.
            if "Measurement_Unit" in df.columns:
                unit_lower = df["Measurement_Unit"].astype(str).str.strip().str.lower()
                qual_mask = unit_lower.isin(["positive", "negative"])
                if qual_mask.any():
                    df.loc[qual_mask, value_col] = unit_lower[qual_mask].map(
                        {"positive": 1.0, "negative": 0.0})
                    logger.info(f"Encoded {int(qual_mask.sum())} qualitative "
                                f"Positive/Negative results as 1/0.")

            # Ensure numeric type for aggregation (handle strings like '3.218e-05' or non-numeric leftovers)
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

            # Permeability: derive Efflux_ratio (BtoA/AtoB) rows. The ratio is
            # dimensionless (both directions share the unit) so it is injected
            # directly without SI conversion.
            if 'Measurement_EffluxRatio' in df.columns:
                ratio_num = pd.to_numeric(df['Measurement_EffluxRatio'], errors='coerce')
                er_mask = ratio_num.notna()
                if er_mask.any():
                    er = df.loc[er_mask].copy()
                    er['gist_endpoint'] = 'Efflux_ratio'
                    er[value_col] = ratio_num.loc[er_mask].values
                    df = pd.concat([df, er], ignore_index=True)
                    logger.info(f"Injected {int(er_mask.sum())} Efflux_ratio rows from Permeability BtoA/AtoB.")

            # Load GIST column list (tab-separated, first line)
            gist_file = os.path.join(os.path.dirname(__file__), "gist", "gist_format.txt")
            with open(gist_file, "r") as gf:
                first_line = gf.readline().strip()
            gist_cols = [c for c in first_line.split("\t") if c]
            if not gist_cols:
                raise RuntimeError("GIST endpoint list is empty. Check gist_format.txt")

            # M1/M2: rows whose endpoint has no GIST target would be silently
            # dropped by the pivot. Instead report them and write to a separate
            # *_unmapped.csv so the loss is visible (e.g. Cytotoxicity,
            # Genetoxicity, pKa/MW, Phase-II metabolism, many VIVO params).
            gist_set = set(gist_cols)
            unmapped_mask = ~df["gist_endpoint"].isin(gist_set)
            if unmapped_mask.any():
                unmapped_df = df.loc[unmapped_mask]
                counts = unmapped_df["endpoint_canonical"].value_counts()
                logger.warning(
                    "%d row(s) across %d endpoint(s) have no GIST target and are "
                    "excluded from the matrix: %s",
                    int(unmapped_mask.sum()), len(counts),
                    ", ".join(f"{ep!r}x{int(n)}" for ep, n in counts.head(20).items()))
                out_dir = "./processed_data"
                os.makedirs(out_dir, exist_ok=True)
                um_base = os.path.basename(args.input_path).split('.')[0]
                um_path = os.path.join(out_dir, f"{um_base}_unmapped.csv")
                unmapped_df.to_csv(um_path, index=False)
                logger.warning("Wrote %d unmapped rows to %s", len(unmapped_df), um_path)

            # Detect PK rows (patient-origin) heuristics
            def is_pk_row(row: pd.Series) -> bool:
                txt = " ".join([str(row.get(col, "")) for col in ["Test", "Test_Type", "Measurement_Type", "Measurment_Type", "Test_Subject"]]).lower()
                return ("patient" in txt) or ("pk" in txt)

            pk_mask = df.apply(is_pk_row, axis=1)
            has_pk = bool(pk_mask.any())

            # ADMET: aggregate mean per SMILES x endpoint
            admet_df = df.loc[~pk_mask].copy()
            admet_matrix = None
            if not admet_df.empty:
                agg = admet_df.groupby([smiles_col, "gist_endpoint"], dropna=False)[value_col].mean().reset_index()
                admet_matrix = agg.pivot_table(index=smiles_col, columns="gist_endpoint", values=value_col, fill_value=0).reset_index()
                # Ensure all GIST columns exist
                for col in gist_cols:
                    if col not in admet_matrix.columns:
                        admet_matrix[col] = 0
                # Order columns: smiles then GIST columns
                admet_matrix = admet_matrix[[smiles_col] + gist_cols]
                admet_matrix = admet_matrix.rename(columns={smiles_col: "smiles"})

            # PK: keep duplicates by expanding rows, one-hot style by endpoint
            pk_df = df.loc[pk_mask].copy()
            pk_matrix = None
            if not pk_df.empty:
                # For each row, build a record with all zeros except the matched endpoint with value
                records = []
                for _, row in pk_df.iterrows():
                    rec = {"smiles": row.get(smiles_col, "")}
                    for col in gist_cols:
                        rec[col] = 0
                    ge = row.get("gist_endpoint", None)
                    if ge in gist_cols:
                        rec[ge] = row.get(value_col, 0) if pd.notna(row.get(value_col, None)) else 0
                    records.append(rec)
                pk_matrix = pd.DataFrame.from_records(records)

            # Concatenate
            if admet_matrix is not None and pk_matrix is not None:
                out_df = pd.concat([admet_matrix, pk_matrix], ignore_index=True)
            elif admet_matrix is not None:
                out_df = admet_matrix
            elif pk_matrix is not None:
                out_df = pk_matrix
            else:
                out_df = pd.DataFrame(columns=["smiles"] + gist_cols)

            # Fill remaining NaNs with 0
            for col in out_df.columns:
                if col != "smiles":
                    out_df[col] = out_df[col].fillna(0)

            # Determine output path
            base_name = os.path.basename(args.input_path).split('.')[0]
            suffix = "_PK" if has_pk else ""
            out_csv = args.gist_output if args.gist_output else os.path.join(args.output_path, f"{base_name}{suffix}_gist_matrix.csv")

            # Save CSV
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            out_df.to_csv(out_csv, index=False)

            # Save metadata
            metadata = {
                "input_path": args.input_path,
                "output_csv": out_csv,
                "rows": int(len(out_df)),
                "columns": [c for c in out_df.columns],
                "has_pk": has_pk,
                "value_column": value_col,
                "endpoint_map_size": int(len(endpoint_map)),
                "endpoint_mapper": mapping_strategy,
            }
            if mapping_strategy == "manual":
                metadata.update({
                    "manual_min_similarity": float(manual_config.min_similarity),
                    "manual_mapping_path": manual_config.mapping_path,
                })
            meta_path = out_csv.replace('.csv', '_metadata.json')
            with open(meta_path, 'w', encoding='utf-8') as mf:
                json.dump(metadata, mf, indent=2, ensure_ascii=False)

            logger.info(f"Saved GIST matrix to {out_csv}")
            logger.info(f"Saved metadata to {meta_path}")
        except Exception as e:
            logger.error(f"GIST matrix conversion failed: {e}")
            raise
        finally:
            logger.info("Finished GIST matrix conversion")
        # Exit after matrix export
        sys.exit(0)
    
    try:
        # Check if input file exists
        if not os.path.exists(args.input_path):
            raise FileNotFoundError(f"Input file not found: {args.input_path}")
        
        # Create output directory at the very beginning
        logger.info(f"Attempting to create output directory: {args.output_path}")
        try:
            os.makedirs(args.output_path, exist_ok=True)
            logger.info(f"Successfully created/verified output directory: {args.output_path}")
        except Exception as dir_error:
            logger.error(f"Failed to create output directory: {dir_error}")
            raise
        
        # Inspect data
        inspector = DataInspector(
            input_path=args.input_path,
            smiles_column=args.smiles_col,
            activity_column=args.activity_col
        )

        # Human PK is a wide-format, non-GIST schema with no Test/Test_Type task
        # grouping; write it out separately rather than crash in quorum grouping.
        if getattr(inspector, "is_human_pk", False):
            os.makedirs(args.output_path, exist_ok=True)
            base = os.path.splitext(os.path.basename(args.input_path))[0]
            hpk_path = os.path.join(args.output_path, f"{base}_HumanPK.csv")
            inspector.df.to_csv(hpk_path, index=False)
            logger.warning(
                "Human PK detected: wrote raw wide-format table to %s and skipped "
                "task-group processing (not a standard long-format schema).", hpk_path)
            sys.exit(0)

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
            os.makedirs(args.output_path, exist_ok=True)
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
                    scale_activity=args.scale_activity,
                    convert_units=args.convert_units,
                    correct_pH=args.correct_pH,
                    pH_method=args.pH_method,
                    target_pH=args.target_pH
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
                
                # Verify output directory exists before saving
                if not os.path.exists(args.output_path):
                    logger.error(f"Output directory does not exist: {args.output_path}")
                    raise FileNotFoundError(f"Output directory does not exist: {args.output_path}")
                
                output_file = os.path.join(args.output_path, f'{base_name}_{task_str}_processed.csv')
                logger.info(f"Attempting to save file to: {output_file}")
                try:
                    processed_data.to_csv(output_file, index=False)
                    logger.info(f"Successfully saved processed data to {output_file}")
                except Exception as save_error:
                    logger.error(f"Failed to save file {output_file}: {save_error}")
                    raise
                
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
                    
                    split_dir = os.path.join(args.output_path, "splits")
                    os.makedirs(split_dir, exist_ok=True)
                    logger.info(f"Created splits directory: {split_dir}")
                    
                    for split_name, split_df in split_data.items():
                        if not split_df.empty:
                            split_file = os.path.join(split_dir, f"{base_name}_{task_str}_{split_name}.csv")
                            split_df.to_csv(split_file, index=False)
                            logger.info(f"Saved {split_name} split to {split_file} ({len(split_df)} samples)")
                
                if args.visualize and not processed_data.empty:
                    viz_dir = os.path.join(args.output_path, "visualizations")
                    os.makedirs(viz_dir, exist_ok=True)
                    logger.info(f"Created visualizations directory: {viz_dir}")
                    
                    visualizer = DataVisualizer(
                        processed_data, 
                        smiles_col='Standardized_SMILES',
                        activity_col=args.activity_col,
                        scale_activity=args.scale_activity
                    )
                    
                    try:
                        mol_viz_path = os.path.join(viz_dir, f"{base_name}_{task_str}_molecules.png")
                        visualizer.visualize_molecules(n_mols=10, output_path=mol_viz_path)
                        
                        act_dist_path = os.path.join(viz_dir, f"{base_name}_{task_str}_activity_dist.png")
                        visualizer.plot_activity_distribution(output_path=act_dist_path)
                        
                        scaffold_path = os.path.join(viz_dir, f"{base_name}_{task_str}_scaffolds.png")
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
