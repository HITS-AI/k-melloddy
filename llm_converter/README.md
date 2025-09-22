# LLM ADME/T Endpoint Converter

A sophisticated tool for converting K-MELLODDY standard data format to GIST format through advanced LLM API integration.
Specialized for intelligent natural language matching between ADME/T (Absorption, Distribution, Metabolism, Excretion, Toxicity) endpoints using pharmaceutical research expertise.

## Features

- **Intelligent ADME/T Endpoint Matching**: Advanced natural language processing with pharmaceutical research expertise
- **Multi-format Support**: Reading K-MELLODDY standard format data (Excel, CSV)
- **LangChain Integration**: Seamless integration with LangChain for LLM provider flexibility
- **Gemini API Support**: Default support for Google's Gemini models with OpenAI fallback
- **Expert-level Matching**: Senior pharmaceutical researcher-level understanding of ADMET processes
- **Comprehensive Output**: GIST format (CSV + metadata) with confidence scores and reasoning
- **Robust Error Handling**: Advanced error handling and comprehensive logging
- **Species-aware Matching**: Understanding of species differences (human, rat, mouse, etc.)
- **Methodology Recognition**: Distinction between in vitro vs in vivo assays

## Installation

### Prerequisites

- Python 3.8+
- Conda environment (recommended: goldilocks)

### Install Dependencies

```bash
# Activate conda environment
conda activate goldilocks

# Install required packages
pip install -r requirements.txt
```

### Required Packages

The following packages are automatically installed:
- `langchain>=0.1.0`: LangChain framework for LLM integration
- `langchain-openai>=0.1.0`: OpenAI-compatible LLM providers (including Gemini)
- `langchain-core>=0.1.0`: Core LangChain functionality
- `pandas`: Data manipulation
- `openpyxl`: Excel file support
- `requests`: HTTP requests

## Configuration

### Environment Variables

Set the following environment variables for LLM configuration:

```bash
# Required: Gemini API Key
export GEMINI_API_KEY="your_gemini_api_key_here"

# Optional: LLM Model Configuration
export LLM_MODEL="gemini-1.5-flash"          # Default model
export LLM_TEMPERATURE="0.1"                  # Low temperature for consistent results
export LLM_SOURCE="Gemini"                    # Default to Gemini, fallback to OpenAI
```

### API Key Setup

1. **Gemini API Key**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. **OpenAI API Key** (fallback): Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)

### Configuration Options

- **Model Selection**: Choose between Gemini models (`gemini-1.5-flash`, `gemini-1.5-pro`) or OpenAI models (`gpt-3.5-turbo`, `gpt-4`)
- **Temperature**: Lower values (0.1-0.3) provide more consistent results for scientific matching
- **Source**: `Gemini` (default) or `OpenAI` for different LLM providers

## Usage

### Basic Usage

```python
from llm_converter.src.format_converter import LLMFormatConverter, ConversionConfig

# Configuration with Gemini API
config = ConversionConfig(
    api_key="your_gemini_api_key",
    model_name="gemini-1.5-flash",
    temperature=0.1,
    source="Gemini"  # Default to Gemini
)

# Initialize converter
converter = LLMFormatConverter(config)

# Load data and convert ADME/T endpoint matching
df = converter.load_k_melloddy_data("input_data/sample.xlsx")
converted_df = converter.convert_to_gist_format(df)
converter.save_gist_format(converted_df, "output_data/gist_output.csv")
```

### Advanced Usage with OpenAI Fallback

```python
# Configuration with OpenAI fallback
config = ConversionConfig(
    api_key="your_openai_api_key",
    model_name="gpt-3.5-turbo",
    temperature=0.1,
    source="OpenAI"
)

converter = LLMFormatConverter(config)
```

### Command Line Usage

```bash
python llm_converter/src/format_converter.py
```

## ADME/T Endpoint Matching

### Expert-level Matching Capabilities

The converter uses a sophisticated prompt system that simulates a **senior pharmaceutical researcher** with extensive expertise in:

- **ADMET Assay Methodologies**: Understanding of various experimental approaches and their biological significance
- **Naming Conventions**: Recognition of different terminology used across pharmaceutical companies and research institutions
- **Biological Process Understanding**: Ability to identify when different endpoint names refer to the same underlying biological process
- **Species Differences**: Understanding of human, rat, mouse, and other species variations and their impact on data interpretation
- **Methodology Recognition**: Distinction between in vitro vs in vivo assays and their relevance

### Advanced Matching Examples

| K-MELLODDY Endpoint | GIST Endpoint | Confidence | Matching Reason |
|---------------------|---------------|------------|-----------------|
| Caco-2 cell permeability | Caco2 | 0.99 | Direct match for Caco-2 monolayer permeability |
| Plasma protein binding percentage | PPBR | 0.98 | Same biological process, different expression |
| Mouse liver microsomal stability | HLM | 0.98 | Species difference but same metabolic process |
| CYP3A4 time-dependent inhibition | CYP3A4_Inhibitor | 0.98 | Core inhibition process with temporal specificity |
| MDCK cell permeability | Caco2 | 0.98 | Different cell lines but same permeability concept |
| Human hepatocyte intrinsic clearance | Clearance_Hepatocyte_AZ | 0.98 | Same clearance measurement with method specification |
| P-gp substrate | pgp_substrate | 0.99 | Identical biological process, naming convention difference |
| Volume of distribution at steady state | VDss | 0.99 | Standard pharmacokinetic parameter abbreviation |

### Complex Matching Scenarios

The system excels at handling:

1. **Species Variations**: Mouse vs Human vs Rat endpoints
2. **Methodology Differences**: Different cell lines (Caco-2 vs MDCK) for same process
3. **Terminology Variations**: Different naming conventions for identical processes
4. **Regulatory vs Research**: Different contexts for same measurements
5. **Temporal Specificity**: Time-dependent vs general inhibition
6. **Unit Conversions**: Different units for same measurement type

### Column Conversion

| K-MELLODDY | GIST | Description |
|------------|------|-------------|
| Chemical ID | compound_id | Compound identifier |
| Chemical Name | compound_name | Compound name |
| SMILES_Structure_Parent | smiles | SMILES structure |
| Test | original_endpoint | Original endpoint |
| Test | gist_endpoint | Matched GIST endpoint |
| Measurement_Value | activity_value | Activity value |
| Measurement_Unit | activity_unit | Activity unit |

## Testing

### Unit Tests

```bash
# Run all tests
python -m pytest llm_converter/tests/

# Run specific test file
python llm_converter/tests/test_converter.py
```

### Integration Testing

The converter has been thoroughly tested in the `goldilocks` conda environment with:

- ✅ **Basic LLM API calls** with Gemini
- ✅ **Endpoint matching** with complex ADMET scenarios
- ✅ **Full data conversion pipeline** from K-MELLODDY to GIST format
- ✅ **Error handling** for various edge cases
- ✅ **JSON parsing** with markdown code block handling
- ✅ **Column mapping** for different data formats

### Test Coverage

- LangChain integration with ChatOpenAI
- Gemini API connectivity and response handling
- Endpoint matching accuracy with confidence scores
- Data format conversion and validation
- Error handling and logging functionality

## Output Files

### Generated Files

- **`output.csv`**: Converted GIST format data with matched endpoints
- **`output_metadata.json`**: Conversion metadata including:
  - Endpoint mapping results
  - Confidence scores
  - Matching reasoning
  - Processing statistics

### Output Format

The converted CSV includes:
- Original compound data
- Matched GIST endpoints
- Confidence scores for each match
- Detailed reasoning for matching decisions

## Logging

### Log Files

The conversion process is logged to:
- **`logs/conversion.log`**: Detailed conversion process logs
- **Console output**: Real-time progress and status updates

### Log Levels

- **INFO**: General process information
- **WARNING**: Non-critical issues (e.g., JSON parsing fallbacks)
- **ERROR**: Critical errors requiring attention

## Notes

### Key Features

- **Gemini Integration**: Uses Gemini as the default LLM provider through LangChain's ChatOpenAI
- **Expert-level Matching**: Sophisticated prompt system simulating senior pharmaceutical researcher expertise
- **Multi-provider Support**: Supports both Gemini and OpenAI models with automatic fallback
- **Robust Error Handling**: Advanced JSON parsing with markdown code block handling
- **Flexible Data Formats**: Supports various K-MELLODDY input formats and column structures

### Important Considerations

- **API Costs**: LLM API usage may incur costs based on token consumption
- **Rate Limits**: Consider API rate limits for large datasets
- **Data Security**: Keep API keys secure and never commit them to version control
- **Environment**: Recommended to use `goldilocks` conda environment for optimal compatibility
- **Batch Processing**: Adjust batch size for large datasets to optimize performance

### Performance Tips

- Use lower temperature values (0.1-0.3) for consistent scientific matching
- Monitor API usage and costs during large-scale conversions
- Enable detailed logging for debugging and optimization
- Test with small datasets before processing large files

## License

MIT License
