# K-MELLODDY
## Overview
This repository contains a Python-based preprocessing pipeline for K-MELLODDY standard data format, primarily designed for tasks in ADME/T prediction. The pipeline supports SMILES standardization, outlier detection, feature scaling, label creation, and now includes multiprocessing support, data visualization, and machine learning-ready data splitting. It is suitable for classification and regression tasks.
This repository will be updated when the K-MELLODDY standard data format is changed.

## Features
- **SMILES Standardization**:
  - Removes salts, isotopes, and stereochemistry (optional).
  - Standardizes tautomeric forms and calculates molecular scaffolds.
  - Parallel processing for large datasets.
- **Label Processing**:
  - Handles binary, categorical, and continuous labels.
  - Converts continuous labels into classification labels using thresholds.
  - Scales experimental values using `StandardScaler`.
- **Outlier Detection**:
  - Supports multiple methods for outlier detection, including IQR, Local Outlier Factor (LOF), One-Class SVM, and Gaussian Mixture Models (GMM).
- **Task Validation**:
  - Ensures the dataset aligns with the selected task (classification or regression).
  - Automatically detects appropriate task type based on data characteristics.
  - Validates training quorum requirements for each test group.
- **Data Visualization**:
  - Visualizes molecular structures.
  - Plots activity value distributions.
  - Displays scaffold diversity analysis.
- **Data Splitting**:
  - Supports random, scaffold-based, and stratified splitting methods.
  - Prepares data for machine learning with train/validation/test sets.
- **Advanced Logging**:
  - Comprehensive logging to both console and file.
  - Detailed tracking of training quorum validation.
  - Timestamped logs for process tracking and debugging.
- **Customizable Parameters**:
  - Command-line interface for all major options.
  - Options to retain stereochemistry, remove salts, detect outliers, and handle duplicates.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HITS-AI/k-melloddy.git
   cd k-melloddy
   ```
2. Install the required dependencies: (will be added soon)
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Command Line Usage
```bash
python hits-preprocess.py --input_path data/chemical_data.csv --output_path ./processed_data --visualize True --parallel True --split scaffold --activity_col "Measurement_Value" --debug
```

### Key Command Line Arguments
| Argument | Description |
|----------|-------------|
| `--input_path` | Path to the input CSV file (required) |
| `--output_path` | Directory to save processed data (default: ./processed_data) |
| `--visualize` | Generate visualizations (True/False) |
| `--parallel` | Use parallel processing for large datasets (True/False) |
| `--scale_activity` | Scale activity values (True/False) |
| `--split` | Data splitting method (random, scaffold, stratified) |
| `--test_size` | Test set size (default: 0.2) |
| `--valid_size` | Validation set size (default: 0.1) |
| `--activity_col` | Name of activity value column (default: Measurement_Value) |
| `--smiles_col` | Name of SMILES column (default: SMILES_Structure_Parent) |
| `--debug` | Enable debug level logging |

### Output Structure
The processed data will be saved in the following structure:
```
output_path/
├── logs/                           # Log files with timestamp
├── splits/                         # Train/validation/test splits
│   ├── dataset_task_train.csv
│   ├── dataset_task_valid.csv
│   └── dataset_task_test.csv
├── visualizations/                 # Visualization files
│   ├── dataset_task_molecules.png  # Molecular structure visualization
│   ├── dataset_task_activity_dist.png  # Activity distribution plots
│   └── dataset_task_scaffolds.png  # Scaffold diversity visualization
└── dataset_task_processed.csv      # Processed data for each task
```

### Input File Format
The input file should be a CSV file with at least two columns:
- **SMILES Column**: Contains SMILES strings of compounds (default: `SMILES_Structure_Parent`).
- **Activity Column**: Contains numeric or categorical activity values (default: `Measurement_Value`).

### Training Quorum Requirements
The pipeline enforces the following training quorum requirements:
- **Classification tasks**: At least 25 active and 25 inactive samples per task.
- **Regression tasks**: At least 50 total data points with at least 25 uncensored data points per task.

## Classes
### DataInspector
Examines the input data, validates training quorum requirements, and organizes data by test groups.

### Preprocessor
Handles chemical standardization, outlier detection, and label preprocessing.

### DataVisualizer
Provides visualization tools for molecular structures, activity distributions, and scaffold diversity.

### DataSplitter
Implements data splitting methods for preparing machine learning datasets.

## Dependencies
- `pandas`
- `numpy`
- `rdkit`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `multiprocessing`
  
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [RDKit](https://www.rdkit.org/): For chemical informatics and machine learning tools.
- [scikit-learn](https://scikit-learn.org/): For machine learning algorithms and preprocessing utilities.

## Contact
For questions or issues, please open an issue on the repository or contact the maintainer at `jhjeon@hits.ai`.
