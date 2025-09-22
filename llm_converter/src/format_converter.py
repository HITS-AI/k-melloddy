#!/usr/bin/env python3
"""
Module for converting K-MELLODDY standard data format to GIST format through LLM API

This module converts through natural language matching between ADME/T endpoints:
1. Reading K-MELLODDY standard format data
2. ADME/T endpoint matching through LLM API
3. Output in GIST format
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConversionConfig:
    """Format conversion configuration"""
    api_key: str
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 30
    source: str = "Gemini"  # Default to Gemini

class LLMFormatConverter:
    """ADME/T endpoint matching converter using LLM API"""
    
    def __init__(self, config: ConversionConfig):
        self.config = config
        self.llm = self._create_llm()
        self.gist_endpoints = self.load_gist_endpoints()
    
    def _create_llm(self) -> ChatOpenAI:
        """Create LLM instance based on configuration"""
        if self.config.source == "Gemini":
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                api_key=self.config.api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                max_tokens=self.config.max_tokens,
            )
        else:
            # Fallback to OpenAI
            return ChatOpenAI(
                model=self.config.model_name,
                temperature=self.config.temperature,
                api_key=self.config.api_key,
                max_tokens=self.config.max_tokens,
            )
    
    def load_gist_endpoints(self) -> List[str]:
        """Load GIST format endpoint list"""
        try:
            gist_file = Path(__file__).parent.parent.parent / "gist" / "gist_format.txt"
            with open(gist_file, 'r') as f:
                endpoints = f.read().strip().split('\t')
            logger.info(f"GIST endpoints loaded: {len(endpoints)} items")
            return endpoints
        except Exception as e:
            logger.error(f"Failed to load GIST endpoints: {e}")
            return []
    
    def load_k_melloddy_data(self, file_path: str) -> pd.DataFrame:
        """Load K-MELLODDY standard format data"""
        try:
            if file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"K-MELLODDY data loaded: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            raise
    
    def create_endpoint_matching_prompt(self, k_melloddy_endpoint: str) -> str:
        """Generate prompt for ADME/T endpoint matching"""
        gist_endpoints_str = ", ".join(self.gist_endpoints)
        
        prompt = f"""
You are an ADME/T (Absorption, Distribution, Metabolism, Excretion, Toxicity) endpoint matching expert.

K-MELLODDY format endpoint: "{k_melloddy_endpoint}"

Please match this to the most appropriate endpoint from the following GIST format endpoints:

{gist_endpoints_str}

Matching rules:
1. Select the most semantically similar endpoint
2. Prioritize endpoints measuring the same biological process
3. Prioritize endpoints using the same tissue/cell type
4. Prioritize endpoints using the same measurement methodology

Example matches:
- "PAMPA permeability" → "PAMPA_pH7.4(bc)" or "PAMPA_pH7.4(mc)"
- "Caco-2 permeability" → "Caco2"
- "CYP3A4 inhibition" → "CYP3A4_Inhibitor"
- "hERG inhibition" → "hERG"
- "Skin permeability" → "skin_permeability"

Response format:
{{
    "matched_endpoint": "exact_GIST_endpoint_name",
    "confidence": 0.95,
    "reasoning": "explanation of matching reason"
}}

Please respond in JSON format only.
"""
        return prompt
    
    def call_llm_api(self, prompt: str) -> str:
        """Call LLM API using langchain"""
        try:
            messages = [
                SystemMessage(content="""You are a senior pharmaceutical researcher with extensive expertise in ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) studies and drug development. You have deep knowledge of:

- Various ADMET assay methodologies and their biological significance
- Different naming conventions used across pharmaceutical companies and research institutions
- How the same biological processes can be described using different terminology
- The underlying mechanisms behind each ADMET endpoint
- Species differences (human, rat, mouse, etc.) and their impact on data interpretation
- In vitro vs in vivo assay differences and their relevance

Your task is to accurately identify when different endpoint names refer to the same underlying biological process or measurement, even when they use different terminology, abbreviations, or experimental conditions. You understand that the same ADMET property can be measured through various methods and expressed in different ways, but the core biological meaning remains consistent.

When matching endpoints, consider:
1. The fundamental biological process being measured
2. The experimental methodology (cell lines, tissues, species)
3. The measurement units and their scientific meaning
4. The context of drug development and regulatory requirements

Provide precise matches based on scientific accuracy, not just superficial similarity."""),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise
    
    def match_endpoints(self, df: pd.DataFrame) -> Dict[str, str]:
        """Match K-MELLODDY endpoints to GIST endpoints"""
        endpoint_mapping = {}
        
        # Extract unique endpoints
        endpoint_column = 'Test' if 'Test' in df.columns else 'endpoint'
        unique_endpoints = df[endpoint_column].unique() if endpoint_column in df.columns else []
        
        for endpoint in unique_endpoints:
            if endpoint not in endpoint_mapping:
                logger.info(f"Matching endpoint: {endpoint}")
                
                # Call LLM API
                prompt = self.create_endpoint_matching_prompt(str(endpoint))
                response = self.call_llm_api(prompt)
                
                try:
                    # Clean response - remove markdown code blocks if present
                    cleaned_response = response.strip()
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]  # Remove ```json
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]  # Remove ```
                    cleaned_response = cleaned_response.strip()
                    
                    # Parse JSON response
                    match_result = json.loads(cleaned_response)
                    matched_endpoint = match_result.get('matched_endpoint', endpoint)
                    confidence = match_result.get('confidence', 0.0)
                    reasoning = match_result.get('reasoning', '')
                    
                    endpoint_mapping[endpoint] = matched_endpoint
                    logger.info(f"Match completed: {endpoint} → {matched_endpoint} (confidence: {confidence})")
                    logger.info(f"Reason: {reasoning}")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing failed for '{endpoint}': {e}")
                    logger.warning(f"Raw response: {response}")
                    endpoint_mapping[endpoint] = endpoint
        
        return endpoint_mapping
    
    def convert_to_gist_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert K-MELLODDY data to GIST format"""
        # 1. Endpoint matching
        endpoint_mapping = self.match_endpoints(df)
        
        # 2. Basic column conversion
        converted_df = df.copy()
        
        # Column name conversion
        column_mapping = {
            'Chemical ID': 'compound_id',
            'Chemical Name': 'compound_name',
            'SMILES_Structure_Parent': 'smiles',
            'Test': 'original_endpoint',
            'Test_Type': 'test_type',
            'Test_Subject': 'test_subject',
            'Measurement_Type': 'measurement_type',
            'Measurement_Value': 'activity_value',
            'Measurement_Unit': 'activity_unit'
        }
        
        # Convert only existing columns
        existing_mapping = {k: v for k, v in column_mapping.items() if k in converted_df.columns}
        converted_df = converted_df.rename(columns=existing_mapping)
        
        # 3. Add GIST endpoint column
        endpoint_col = 'original_endpoint' if 'original_endpoint' in converted_df.columns else 'endpoint'
        if endpoint_col in converted_df.columns:
            converted_df['gist_endpoint'] = converted_df[endpoint_col].map(endpoint_mapping)
            # Replace original endpoint with GIST endpoint
            converted_df[endpoint_col] = converted_df[endpoint_col].map(endpoint_mapping)
        
        # 4. Data type conversion
        if 'activity_value' in converted_df.columns:
            converted_df['activity_value'] = pd.to_numeric(
                converted_df['activity_value'], errors='coerce'
            )
        
        return converted_df
    
    def apply_conversion_rules(self, df: pd.DataFrame, rules: Dict[str, str]) -> pd.DataFrame:
        """Apply conversion rules"""
        converted_df = df.copy()
        
        # Column name conversion
        column_mapping = {
            'Chemical ID': 'compound_id',
            'Chemical Name': 'compound_name',
            'SMILES_Structure_Parent': 'smiles',
            'Test': 'assay_type',
            'Test_Type': 'test_type',
            'Test_Subject': 'test_subject',
            'Measurement_Type': 'measurement_type',
            'Measurement_Value': 'activity_value',
            'Measurement_Unit': 'activity_unit'
        }
        
        converted_df = converted_df.rename(columns=column_mapping)
        
        # Data type conversion
        if 'activity_value' in converted_df.columns:
            converted_df['activity_value'] = pd.to_numeric(
                converted_df['activity_value'], errors='coerce'
            )
        
        return converted_df
    
    def apply_default_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply default conversion rules (when LLM API fails)"""
        return self.apply_conversion_rules(df, {})
    
    def save_gist_format(self, df: pd.DataFrame, output_path: str):
        """Save in GIST format"""
        try:
            # Add metadata
            metadata = {
                'format_version': '1.0',
                'source_format': 'K-MELLODDY',
                'conversion_method': 'LLM_API',
                'total_records': len(df),
                'columns': list(df.columns)
            }
            
            # Save as CSV
            df.to_csv(output_path, index=False)
            
            # Save metadata separately
            metadata_path = output_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"GIST format saved: {output_path}")
            
        except Exception as e:
            logger.error(f"Save failed: {e}")
            raise

def main():
    """Main execution function"""
    # Load configuration
    config = ConversionConfig(
        api_key=os.getenv('GEMINI_API_KEY', ''),
        model_name=os.getenv('LLM_MODEL', 'gemini-1.5-flash'),
        temperature=float(os.getenv('LLM_TEMPERATURE', '0.1')),
        source=os.getenv('LLM_SOURCE', 'Gemini')
    )
    
    if not config.api_key:
        logger.error("GEMINI_API_KEY environment variable is not set.")
        return
    
    # Initialize converter
    converter = LLMFormatConverter(config)
    
    # Input file path
    input_file = "input_data/data_sample.xlsx"  # Use sample data
    
    try:
        # Load K-MELLODDY data
        df = converter.load_k_melloddy_data(input_file)
        logger.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        # ADME/T endpoint matching and conversion
        converted_df = converter.convert_to_gist_format(df)
        
        # Save in GIST format
        output_file = "output_data/gist_format_output.csv"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        converter.save_gist_format(converted_df, output_file)
        
        logger.info("ADME/T endpoint matching and format conversion completed!")
        logger.info(f"Converted data: {len(converted_df)} rows")
        
        # Print matching results summary
        if 'gist_endpoint' in converted_df.columns:
            endpoint_summary = converted_df.groupby(['original_endpoint', 'gist_endpoint']).size().reset_index(name='count')
            logger.info("Endpoint matching results:")
            for _, row in endpoint_summary.iterrows():
                logger.info(f"  {row['original_endpoint']} → {row['gist_endpoint']} ({row['count']} items)")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        raise

if __name__ == "__main__":
    main()
