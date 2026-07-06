# K-MELLODDY
## Overview
This repository contains a Python-based preprocessing pipeline for K-MELLODDY standard data format, primarily designed for tasks in ADME/T prediction. The pipeline supports SMILES standardization, outlier detection, feature scaling, label creation, and now includes multiprocessing support, data visualization, and machine learning-ready data splitting. It is suitable for classification and regression tasks.
This repository will be updated when the K-MELLODDY standard data format is changed.

## K-MELLODDY Standard Format v4.6 Support
The pipeline was updated to fully support the **v4.6** standard data format (previously only ~25–30% was covered, with most v4.6 files producing empty/all-zero matrices or crashes). Highlights of the v4.6 adaptation:

- **Column-name normalization**: case-insensitive canonicalization of `Measurement_*` (incl. the legacy `Measurment_*` typo), `Test_Subject*` (asterisk), `Test_Species → Test_Subject`, and VIVO's lowercase `Measurement_value`. The value column is preserved (the old `Unnamed:10` hard-coding is gone).
- **Permeability**: `Measurement_Value(AtoB)` / `(BtoA)` bracket-list values (`[6,2]`, `[6,]`, `[,2]`) are parsed; AtoB feeds `Caco2`/`PAMPA` and `Efflux_ratio` is derived as `BtoA/AtoB`.
- **Human PK**: wide-format, 3-row-header workbooks (`데이터1/2(Human PK)`) are detected and written to a separate `*_HumanPK.csv` (not a GIST-mappable schema) instead of crashing.
- **pH sources**: pH is inferred from `Test_Subject` (`pH7.4`), `Measurement_Condition` (bare number), and `Test_Type`, in addition to dedicated pH columns.
- **Units**: composite/exponent units are preserved — `10-6 cm/s` is no longer corrupted, molar units are case-sensitive (`uM` ≠ micrometre), and dimensionless units (`ratio`, `fold`, `%`) keep their value. Unparseable units are kept with a warning.
- **Endpoint mapping**: exact CYP-isoform mapping (`CYP1A1`/`CYP2C8` have no GIST slot → left unmapped; `CYP3A4_MDZ`/`CYP3A4_TST` merge into `CYP3A4_Inhibitor`), whole-token matching so `tr`/`gr` no longer leak into `transporter`, and qualitative `Positive`/`Negative` (stored in `Measurement_Unit`) encoded as `1`/`0`.
- **No silent loss**: rows whose endpoint has no GIST column (e.g. Cytotoxicity, Genetoxicity, pKa/MW, Phase-II metabolism) are reported and written to `*_unmapped.csv` (CLI) instead of being dropped silently.

> ⚠️ Core classes (`UnitConverter`, `pHCorrector`, `DataInspector`, `Preprocessor`, and the endpoint mapper) are duplicated across `hits-preprocess.py` (CLI) and `data.py` (Python API). Logic changes must be applied to **both**.

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
  - Infers pH from `Test_Subject` (`pH7.4`), `Measurement_Condition` (bare number), and `Test_Type`, in addition to dedicated pH columns.
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
  - **NEW**: Automatic duplicate column removal to prevent groupby errors.
  - **NEW**: Path traversal protection for secure file loading.
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
  - OmegaConf-backed configuration files with CLI overrides for reproducible runs.
  - Offline-friendly manual endpoint mapping (default mode, no API required).
  - Options to retain stereochemistry, remove salts, detect outliers, and handle duplicates.
  - **NEW**: Export to GIST matrix (SMILES × GIST endpoints) via `--to-gist-matrix`
  - **NEW**: Python API functions (`preprocess_dataframe`, `preprocess_to_gist`) for programmatic use
  - **NEW**: Configurable missing data handling (NaN vs 0) via `fill_missing` parameter

## GIST Matrix Export
This pipeline can convert K-MELLODDY standard data into a GIST format matrix where rows are SMILES and columns are standardized GIST endpoints.

### Key Features
- **Integrated Endpoint Mapping**: Rule-based endpoint matching with 100+ predefined synonyms (fully offline, no API required)
- **Three-Tier Matching Strategy**:
  1. **Exact Match**: Direct lookup in synonym dictionary
  2. **Whole-Token Match**: Contiguous token-sequence matching (prefers the longest, latest match) so short keys like `tr`/`gr` do not substring-match inside `transporter`
  3. **Similarity Match**: Token overlap and sequence similarity (fallback)
- **CYP Endpoint Protection**: A regex-based CYP guard (in **both** mappers) maps only the exact isoform; `CYP1A1`/`CYP2C8` (no GIST slot) stay unmapped and `CYP3A4_MDZ`/`CYP3A4_TST` merge into `CYP3A4_Inhibitor`
- **Unmappable Endpoints**: Endpoints with no GIST column (`Cytotoxicity`, `Genetoxicity`) are never force-mapped via similarity; they are reported to `*_unmapped.csv`
- **SI Unit Conversion**: Automatically converts units to SI using the built-in `UnitConverter`
- **Missing Data Handling**: Uses `NaN` by default to distinguish between actual zero measurements and missing data (configurable via `fill_missing` parameter)
- **PK Data Support**: For PK (patient-origin) data, duplicate rows per SMILES are preserved with aggregated measurements
- **Column Normalization**: Automatic case-insensitive column name normalization for backward compatibility

### Endpoint Mapping Modes
- **`manual`** (default): Uses integrated `ManualFormatConverter` with rule-based matching. Fully offline, no API keys required.
  - 100+ predefined endpoint synonyms covering major ADMET categories
  - Custom mapping files (CSV/JSON) can be provided via `manual_mapping_path`
  - Configurable similarity threshold (`manual_min_similarity`, default: 0.55)
- **`llm`** (optional): Uses `llm_converter/src/format_converter.py` with LangChain + Gemini/OpenAI (requires API keys)
  - Automatically falls back to `manual` mode if credentials or dependencies are missing

### GIST Endpoint List
The list of GIST endpoints is defined in `gist/gist_format.txt` (and mirrored in `data.py`). It contains 86 standardized endpoint columns covering:
- Permeability (Caco2, PAMPA, HIA, BBB, etc.)
- Transporters (P-gp, BCRP, OATP, MATE, OCT2)
- CYPs (CYP1A2, CYP2B6, CYP2C9, CYP2C19, CYP2D6, CYP3A4 as Inhibitor/Substrate)
- Metabolism (HLM, RLM, HLC_Stability, Clearance)
- Toxicity (hERG, AMES, DILI, ClinTox, Micronucleus, etc.)
- Nuclear Receptors (NR-AhR, NR-AR, NR-ER, etc.)

### Usage

#### Using Command Line Interface
```bash
conda activate goldilocks
python hits-preprocess.py \
  --input_path input_data/Des_43404_PAMPA_Caco-2_MDCK_250829.xlsx \
  --output_path . \
  --to-gist-matrix \
  --endpoint_mapper manual \
  --fill_missing false \
  --debug
```

#### Using Python API
```python
from data import preprocess_dataframe, preprocess_to_gist
import pandas as pd

# Step 1: Preprocess long format data
df = pd.read_excel('input_data.xlsx')
preprocessed_df = preprocess_dataframe(
    df=df,
    task_type='regression',
    task='permeability',
    smiles_column='smiles_structure_parent',
    activity_column='measurement_value',
    convert_units=True,
    correct_pH=True
)

# Step 2: Convert to GIST format
gist_matrix = preprocess_to_gist(
    input_data=preprocessed_df,
    skip_preprocessing=True,  # Already preprocessed
    endpoint_mapper='manual',
    fill_missing=False,  # Use NaN for missing values
    manual_min_similarity=0.55
)

# Save result
gist_matrix.to_csv('output_gist_matrix.csv', index=False)
```

### Generated Files
- `*_gist_matrix.csv`: GIST format matrix (SMILES × GIST endpoints)
- `*_PK_gist_matrix.csv`: If PK rows detected (includes aggregated measurements)
- `*_unmapped.csv`: **NEW** — rows whose endpoint has no GIST column (reported instead of silently dropped)
- `*_HumanPK.csv`: **NEW** — Human PK workbooks (wide-format, non-GIST schema) written out separately
- Corresponding `*_gist_matrix_metadata.json`: Processing metadata (if using CLI)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/HITS-AI/k-melloddy.git
   cd K-MELLODDY
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   
   For LLM endpoint mapping (optional):
   ```bash
   # Set environment variables (see GIST Matrix Export section)
   export GEMINI_API_KEY="<your_gemini_api_key>"
   ```

## Usage
### Command Line Usage
```bash
python hits-preprocess.py \
  --config config/preprocess_defaults.yaml \
  --input_path input_data/data_sample.csv \
  --split scaffold \
  --visualize true \
  --parallel true
```

### Configuration (OmegaConf)
- The script loads defaults from `config/preprocess_defaults.yaml`. Copy and modify this file for project-specific presets.
- Pass `--config path/to/your_config.yaml` (or `config=...`) to load another YAML file. Subsequent CLI flags override the file values.
- Arguments still accept `--flag value` syntax; boolean flags can be toggled with `--flag true/false` or `flag=true`.
- To run without a config file, omit `--config` and supply all necessary options on the command line; `--input_path` remains required.
- Choose the endpoint mapping mode via `endpoint_mapper` (`llm` or `manual`); manual mode needs no API key and supports optional custom synonym files.

### Key Command Line Arguments
| Argument | Description |
|----------|-------------|
| `--config` | Path to a YAML config file (e.g. `config/preprocess_defaults.yaml`) |
| `--input_path` | Path to the input file (CSV or Excel) (required) |
| `--output_path` | Directory to save processed data (default: ./processed_data) |
| `--visualize` | Generate visualizations (True/False) |
| `--parallel` | Use parallel processing for large datasets (True/False) |
| `--scale_activity` | Scale activity values (True/False) |
| `--convert_units` | Convert units to SI units (True/False) |
| `--correct_pH` | Correct pH-dependent activity values (True/False) |
| `--pH_method` | pH correction method (all, henderson_hasselbalch, empirical, molecular_properties) |
| `--target_pH` | Target pH for correction (default: 7.4) |
| `--to-gist-matrix` | Export to GIST format matrix (SMILES × GIST endpoints) |
| `--endpoint_mapper` | Endpoint mapping mode (`llm` or `manual`, default: `manual`) |
| `--manual_mapping_path` | Optional CSV/JSON for additional manual endpoint synonyms |
| `--manual_min_similarity` | Minimum similarity threshold for manual mapping (default: 0.55) |
| `--manual_prefer_exact` | Prefer exact synonym hits before similarity (True/False, default: True) |
| `--fill_missing` | Fill missing values with 0 instead of NaN (True/False, default: False) |
| `--skip_preprocessing` | Skip DataInspector normalization (use with preprocessed data) |
| `--split` | Data splitting method (random, scaffold, stratified) |
| `--test_size` | Test set size (default: 0.2) |
| `--valid_size` | Validation set size (default: 0.1) |
| `--activity_col` | Name of activity value column (default: measurement_value) |
| `--smiles_col` | Name of SMILES column (default: smiles_structure_parent) |
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

> Note: The latest K-MELLODDY standard data format and sample files can be downloaded from the official forum: [forum.k-melloddy.com](https://forum.k-melloddy.com/).

Required columns in both formats:
- **SMILES Column**: Contains SMILES strings of compounds (default: `smiles_structure_parent`, case-insensitive matching supported).
- **Activity Column**: Contains numeric or categorical activity values (default: `measurement_value`, case-insensitive matching supported).

Optional but recommended columns:
- **Test_Subject**: Identifies the test subject (e.g., human, rat). In new format, this may be labeled as 'Test_Subject*' (automatically normalized).
- **Test_Dose**: Specifies dosage information (automatically included for Pharmacokinetics tests).
- **Chemical_ID**: Compound identifier (will be included in molecule visualizations if present).
- **Measurement_Unit**: Unit of measurement (e.g., μg/mL, μM, hours) for unit conversion to SI units. Creates `measurement_value_si` and `measurement_unit_si` columns.
- **pH-related columns**: pH value column (e.g., pH, pH_Value, Measurement_pH) for pH correction. **NEW (v4.6)**: pH is also inferred from `Test_Subject` (`pH7.4`), `Measurement_Condition` (bare number), or `Test_Type` when no dedicated pH column exists.
- **Test**: Test identifier (e.g., 'Pharmacokinetics', 'ADMET').
- **Test_Type**: Type of test (e.g., 'pH4.0', 'pH 7.4').
- **Measurement_Type**: Type of measurement.

**NEW**: Additional columns in the v4.6 K-MELLODDY format:
- **Measurement_Conc**: Concentration information for measurements (implicit `uM`).
- **Measurement_Temp**: Temperature information for measurements.
- **Measurement_Class**: Classification information (multi-meaning; used as a group key so different meanings are not merged).
- **Measurement_Condition**: Assay condition (e.g. pH for Permeability/BBB/PPB).
- **Measurement_Route / Measurement_Sex / Measurement_Formulation**: VIVO PK metadata (included in task grouping).
- **Measurement_Value(AtoB) / Measurement_Value(BtoA)**: Permeability directional values (bracket lists).

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

## Classes and Functions

### Main Functions

#### `preprocess_dataframe(df, task_type, task, ...)`
Preprocesses long format data (one measurement per row) using the `Preprocessor` class.

**Parameters:**
- `df`: Input DataFrame in long format
- `task_type`: 'classification' or 'regression'
- `task`: Task identifier
- `smiles_column`: SMILES column name (default: 'smiles_structure_parent')
- `activity_column`: Activity column name (default: 'measurement_value')
- `convert_units`: Convert units to SI (default: True)
- `correct_pH`: Apply pH correction (default: True)
- Additional Preprocessor parameters

**Returns:** Preprocessed DataFrame with standardized SMILES and processed labels

#### `preprocess_to_gist(input_data, ...)`
Converts input data (CSV path or DataFrame) to GIST format matrix.

**Parameters:**
- `input_data`: CSV file path (str) or DataFrame (can be preprocessed by `preprocess_dataframe`)
- `endpoint_mapper`: 'llm' or 'manual' (default: 'manual')
- `skip_preprocessing`: Skip DataInspector normalization if data is already preprocessed (default: False)
- `fill_missing`: Fill missing values with 0 instead of NaN (default: False)
- `manual_mapping_path`: Path to custom endpoint mapping CSV/JSON (optional)
- `manual_min_similarity`: Minimum similarity for manual matching (default: 0.55)
- Additional configuration options

**Returns:** GIST format matrix (DataFrame with SMILES as rows and GIST endpoints as columns)

### Classes

#### `DataInspector`
Examines the input data, validates training quorum requirements, and organizes data by test groups.
- **NEW**: Automatic column name normalization (case-insensitive)
- **NEW**: Duplicate column removal
- **NEW**: Path traversal protection
- **NEW**: Support for multiple Excel sheets (ADMET, PK, 데이터)

#### `Preprocessor`
Handles chemical standardization, outlier detection, label preprocessing, and unit conversion.
- **NEW**: Improved duplicate handling with fallback strategy
- **NEW**: Automatic SI unit conversion re-run on DataFrame restoration
- **NEW**: Enhanced error handling for edge cases

#### `UnitConverter`
Converts measurement units to SI units using the pint package, supporting common chemical/pharmaceutical units.
- Creates new columns with `_si` suffix (e.g., `measurement_value_si`, `measurement_unit_si`)
- Handles comparison operators (>, <) in measurement values

#### `pHCorrector`
Corrects pH-dependent activity values using three methods: Henderson-Hasselbalch equation, empirical correction, and molecular properties-based correction.
- **NEW**: Automatic pH inference from Test_Type column when pH column is missing

#### `ManualFormatConverter`
Rule-based endpoint mapper (integrated from `manual_converter.py`).
- **NEW**: 100+ predefined endpoint synonyms
- **NEW**: Three-tier matching strategy (exact → substring → similarity)
- **NEW**: Special CYP endpoint protection (prevents false positives)
- **NEW**: Configurable similarity threshold and exact match preference
- Supports custom mapping files (CSV/JSON)

#### `DataVisualizer`
Provides visualization tools for molecular structures, activity distributions, and scaffold diversity.

#### `DataSplitter`
Implements data splitting methods for preparing machine learning datasets.

## Dependencies
- `pandas>=2.2.3`
- `numpy>=2.2.6`
- `rdkit>=2025.3.2`
- `scikit-learn>=1.6.1`
- `scipy>=1.15.3`
- `matplotlib>=3.10.3`
- `seaborn>=0.13.2`
- `openpyxl>=3.0.0` (for Excel file support)
- `pint>=0.19.0` (for unit conversion)
- `omegaconf>=2.3.0` (for YAML/CLI configuration management)

### Optional Dependencies (for LLM endpoint mapping)
- `langchain>=0.1.0`
- `langchain-openai>=0.1.0`
- `langchain-core>=0.1.0`

> **Note**: LLM endpoint mapping is optional. The default `manual` mode works fully offline without any API keys.
  
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- [RDKit](https://www.rdkit.org/): For chemical informatics and machine learning tools.
- [scikit-learn](https://scikit-learn.org/): For machine learning algorithms and preprocessing utilities.

## Contact
For questions or issues, please open an issue on the repository or contact the maintainer at `jhjeon@hits.ai`.
