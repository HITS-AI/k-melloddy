# k-melloddy
# Preprocessor for Chemical Data

## Overview
This repository contains a Python-based preprocessing pipeline for K-MELLODDY standard data format, primarily designed for tasks in ADME/T prediction. The pipeline supports SMILES standardization, outlier detection, feature scaling, and label creation, making it suitable for classification and regression tasks.

## Features
- **SMILES Standardization**:
  - Removes salts, isotopes, and stereochemistry (optional).
  - Standardizes tautomeric forms and calculates molecular scaffolds.
- **Label Processing**:
  - Handles binary, categorical, and continuous labels.
  - Converts continuous labels into classification labels using thresholds.
  - Scales experimental values using `StandardScaler`.
- **Outlier Detection**:
  - Supports multiple methods for outlier detection, including IQR, Local Outlier Factor (LOF), One-Class SVM, and Gaussian Mixture Models (GMM).
- **Task Validation**:
  - Ensures the dataset aligns with the selected task (classification or regression).
- **Customizable Parameters**:
  - Options to retain stereochemistry, remove salts, detect outliers, and handle duplicates.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HITS-AI/k-melloddy.git
   cd k-melloddy
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Example Code
```python
from preprocessor import Preprocessor

# Initialize the Preprocessor
preprocessor = Preprocessor(
    input_path='data/chemical_data.csv',
    task='classification',
    task_name='solubility',
    smiles_column='SMILES',
    activity_column='Activity',
    remove_salt=True,
    keep_stereo=False,
    keep_duplicates=False,
    detect_outliers=True,
    threshold=50
)

# Run Preprocessing
processed_data = preprocessor.preprocess()

# Save the processed data
processed_data.to_csv('data/processed_chemical_data.csv', index=False)
```

### Input File Format
The input file should be a CSV file with at least two columns:
- **SMILES Column**: Contains SMILES strings of compounds (default: `SMILES_Structure_Parent`).
- **Activity Column**: Contains numeric or categorical activity values (default: `Measurement_Value`).

### Parameters
| Parameter         | Type    | Description                                                                                                   |
|-------------------|---------|---------------------------------------------------------------------------------------------------------------|
| `input_path`      | `str`   | Path to the input CSV file.                                                                                  |
| `task`            | `str`   | Task type: `classification` or `regression`.                                                                |
| `task_name`       | `str`   | Name of the task (e.g., `solubility`, `cyp1a2 inhibition`).                                                  |
| `smiles_column`   | `str`   | Name of the column containing SMILES strings.                                                               |
| `activity_column` | `str`   | Name of the column containing activity values.                                                               |
| `remove_salt`     | `bool`  | Whether to remove salts from SMILES strings.                                                                |
| `keep_stereo`     | `bool`  | Whether to retain stereochemistry in SMILES strings.                                                        |
| `keep_duplicates` | `bool`  | Whether to keep duplicate entries.                                                                           |
| `detect_outliers` | `bool`  | Whether to detect and remove outliers.                                                                      |
| `threshold`       | `float` | Threshold for converting continuous labels into classification labels (required for classification tasks).  |

## Methods
### Key Methods
- `preprocess()`: Runs the complete preprocessing pipeline.
- `preprocess_compound(smiles)`: Processes a single SMILES string.
- `detect_outliers_statistical()`: Identifies outliers using the Interquartile Range (IQR) method.
- `detect_outliers_density_based()`: Identifies outliers using Local Outlier Factor (LOF).
- `detect_outliers_classification_based()`: Identifies outliers using One-Class SVM.
- `scale_experiment_values(labels)`: Scales numeric activity values.

## Dependencies
- `pandas`
- `numpy`
- `rdkit`
- `scikit-learn`
- `scipy`
  
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [RDKit](https://www.rdkit.org/): For chemical informatics and machine learning tools.
- [scikit-learn](https://scikit-learn.org/): For machine learning algorithms and preprocessing utilities.

## Contact
For questions or issues, please open an issue on the repository or contact the maintainer at `jhjeon@hits.ai`.
