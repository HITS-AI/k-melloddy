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
- **Unit Conversion**:
  - **NEW**: Automatically converts units to SI (International System of Units) using the `pint` package.
  - Supports common chemical/pharmaceutical units (μg/mL, μM, hours, etc.).
  - Creates new columns with SI-converted values and units.
  - Provides detailed conversion summaries and statistics.
  - Handles comparison operators (>, <) in measurement values.
- **pH Correction**:
  - **NEW**: Corrects pH-dependent activity values to a target pH (default: 7.4, physiological pH).
  - Supports three correction methods: Henderson-Hasselbalch equation, empirical correction, and molecular properties-based correction.
  - Automatically detects pH-related data when 'pH' is found in Test_Type column.
  - Creates new columns with pH-corrected values for each method.
  - Handles missing pH data gracefully.
- **Input File Support**:
  - Supports both CSV and Excel (.xlsx/.xls) file formats.
  - Automatically detects and processes ADMET and PK sheets in Excel files.
  - **NEW**: Supports new K-MELLODDY format with '데이터' sheet (header at row 2).
  - Combines data from multiple sheets for comprehensive analysis.
  - **NEW**: Automatic column name normalization for backward compatibility.
- **Flexible Data Handling**:
  - Automatically includes Test_Dose column for Pharmacokinetics test types.
  - Groups data by Test, Test_Type, Test_Subject, and Measurement_Type for more precise analysis.
  - Preserves Chemical ID information throughout the processing pipeline.
  - **NEW**: Special handling for Pharmacokinetics data - preserves duplicate compounds with different measurements.
  - **NEW**: Improved duplicate handling for all data types, with fallback to keep='first' strategy if all data would be removed.
- **Special Character Handling**:
  - **NEW**: Automatically replaces special characters (μ, °, α, β, etc.) with their ASCII equivalents to prevent encoding issues.
  - Ensures data compatibility across different systems and software.
- **Outlier Detection**:
  - Supports multiple methods for outlier detection, including IQR, Local Outlier Factor (LOF), One-Class SVM, and Gaussian Mixture Models (GMM).
- **Task Validation**:
  - Ensures the dataset aligns with the selected task (classification or regression).
  - Automatically detects appropriate task type based on data characteristics.
  - Validates training quorum requirements for each test group.
  - **NEW**: Enhanced task validation using valid_options.csv for reference validation.
- **Data Visualization**:
  - Visualizes molecular structures with activity values and Chemical IDs.
  - Plots activity value distributions.
  - Displays scaffold diversity analysis.
  - **NEW**: Improved handling of duplicate SMILES in visualizations, preserving data integrity.
  - **NEW**: Enhanced legend spacing for better readability of Chemical IDs and activity values.
- **Data Splitting**:
  - Supports random, scaffold-based, and stratified splitting methods.
  - Prepares data for machine learning with train/validation/test sets.
- **Advanced Logging**:
  - Comprehensive logging to both console and file.
  - Detailed tracking of training quorum validation.
  - Timestamped logs for process tracking and debugging.
  - **NEW**: Improved error handling with informative messages, especially for empty datasets.
- **Robust File Naming**:
  - **NEW**: Enhanced file naming logic that properly handles special characters and complex task structures.
  - Ensures valid filenames across different operating systems.
- **Customizable Parameters**:
  - Command-line interface for all major options.
  - Options to retain stereochemistry, remove salts, detect outliers, and handle duplicates.
  - **NEW**: Export to GIST matrix (SMILES × GIST endpoints) via `--to-gist-matrix`

## New: GIST Matrix Export (LLM-assisted)
This pipeline can convert K-MELLODDY standard data into a GIST matrix where rows are SMILES and columns are GIST endpoints.

- Endpoint matching is performed by an LLM (Gemini via LangChain) using domain-specific prompts.
- Units are converted to SI using the built-in `UnitConverter`.
- For ADMET data, duplicate records per SMILES/endpoints are averaged.
- For PK (patient-origin) data, duplicate rows per SMILES are kept, and output filename includes `_PK`.
- Missing values are filled with 0. Metadata JSON is saved alongside the CSV.

### Requirements for LLM mapping
Set the following environment variables before running:

```bash
export GEMINI_API_KEY="<your_gemini_api_key>"
export LLM_MODEL="gemini-1.5-flash"
export LLM_TEMPERATURE="0.1"
export LLM_SOURCE="Gemini"
```

### GIST Endpoint List
The list of GIST endpoints (columns) is read from `K-MELLODDY/gist/gist_format.txt` (first line, tab-separated).

### Usage
Run from the `K-MELLODDY` directory:

```bash
conda activate goldilocks
python hits-preprocess.py \
  --input_path input_data/Des_43404_PAMPA_Caco-2_MDCK_250829.xlsx \
  --output_path . \
  --to-gist-matrix \
  --debug
```

Generated files:
- `*_gist_matrix.csv` (or `*_PK_gist_matrix.csv` if PK rows detected)
- Corresponding `*_gist_matrix_metadata.json`

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
python hits-preprocess.py --input_path input_data/data_sample.csv --output_path ./processed_data --visualize True --parallel True --split scaffold --activity_col "Measurement_Value"
```

### Key Command Line Arguments
| Argument | Description |
|----------|-------------|
| `--input_path` | Path to the input file (CSV or Excel) (required) |
| `--output_path` | Directory to save processed data (default: ./processed_data) |
| `--visualize` | Generate visualizations (True/False) |
| `--parallel` | Use parallel processing for large datasets (True/False) |
| `--scale_activity` | Scale activity values (True/False) |
| `--convert_units` | Convert units to SI units (True/False) |
| `--correct_pH` | Correct pH-dependent activity values (True/False) |
| `--pH_method` | pH correction method (all, henderson_hasselbalch, empirical, molecular_properties) |
| `--target_pH` | Target pH for correction (default: 7.4) |
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
The pipeline supports two main file formats:

#### CSV Files
- Simple comma-separated values format with headers.
- Must contain at least the SMILES column and activity column.

#### Excel Files (.xlsx/.xls)
- Automatically processes 'ADMET' and 'PK' sheets if available.
- **NEW**: Supports new K-MELLODDY format with '데이터' sheet (header at row 2).
- Combines data from both sheets for comprehensive analysis.
- Falls back to the first sheet if specific named sheets aren't found.

Required columns in both formats:
- **SMILES Column**: Contains SMILES strings of compounds (default: `SMILES_Structure_Parent`).
- **Activity Column**: Contains numeric or categorical activity values (default: `Measurement_Value`).

Optional but recommended columns:
- **Test_Subject**: Identifies the test subject (e.g., human, rat). In new format, this may be labeled as 'Test_Subject*'.
- **Test_Dose**: Specifies dosage information (automatically included for Pharmacokinetics tests).
- **Chemical_ID**: Compound identifier (will be included in molecule visualizations if present).
- **Measurement_Unit**: Unit of measurement (e.g., μg/mL, μM, hours) for unit conversion to SI units.
- **pH-related columns**: pH value column (e.g., pH, pH_Value, Measurement_pH) for pH correction when Test_Type contains 'pH'.

**NEW**: Additional columns in new K-MELLODDY format:
- **Measurement_Conc**: Concentration information for measurements.
- **Measurement_Temp**: Temperature information for measurements.
- **Measurement_Class**: Classification information for measurements.

### Special Cases Handling
The pipeline includes special handling for certain data types:

#### Pharmacokinetics Data
- **Duplicate Preservation**: For data with 'Pharmacokinetics' in the Test column, duplicate SMILES are preserved to maintain multiple measurement values for the same compound.
- **Test_Dose Column**: Automatically included for Pharmacokinetics data, with a default value of "Unknown" if not present in the input.

#### Character Encoding
- **Special Character Replacement**: Unicode characters like 'μ' (micro), '°' (degree), Greek letters, and other special symbols are automatically replaced with ASCII equivalents to prevent encoding issues.

### Training Quorum Requirements
The pipeline enforces the following training quorum requirements:
- **Classification tasks**: At least 25 active and 25 inactive samples per task.
- **Regression tasks**: At least 50 total data points with at least 25 uncensored data points per task.

## Classes
### DataInspector
Examines the input data, validates training quorum requirements, and organizes data by test groups.

### Preprocessor
Handles chemical standardization, outlier detection, label preprocessing, and unit conversion.

### UnitConverter
Converts measurement units to SI units using the pint package, supporting common chemical/pharmaceutical units.

### pHCorrector
Corrects pH-dependent activity values using three methods: Henderson-Hasselbalch equation, empirical correction, and molecular properties-based correction.

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
- `openpyxl` (for Excel file support)
- `pint` (for unit conversion)
  
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [RDKit](https://www.rdkit.org/): For chemical informatics and machine learning tools.
- [scikit-learn](https://scikit-learn.org/): For machine learning algorithms and preprocessing utilities.

## Contact
For questions or issues, please open an issue on the repository or contact the maintainer at `jhjeon@hits.ai`.
