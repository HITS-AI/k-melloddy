#!/usr/bin/env python3
"""
LLM format converter test module
"""

import unittest
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch
import sys

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from format_converter import LLMFormatConverter, ConversionConfig

class TestLLMFormatConverter(unittest.TestCase):
    """LLM format converter test"""
    
    def setUp(self):
        """Test setup"""
        self.config = ConversionConfig(
            api_key="test_key",
            model_name="gemini-1.5-flash",
            source="Gemini"
        )
        self.converter = LLMFormatConverter(self.config)
        
        # Sample test data
        self.sample_data = pd.DataFrame({
            'Chemical ID': ['C001', 'C002'],
            'Chemical Name': ['Test Compound 1', 'Test Compound 2'],
            'SMILES_Structure_Parent': ['CCO', 'CCN'],
            'Test': ['Permeability', 'Permeability'],
            'Test_Type': ['Caco-2', 'Caco-2'],
            'Test_Subject': ['Human', 'Human'],
            'Measurement_Type': ['Permeability', 'Permeability'],
            'Measurement_Value': [1.5, 2.3],
            'Measurement_Unit': ['10-6 cm/s', '10-6 cm/s']
        })
    
    def test_load_k_melloddy_data_csv(self):
        """CSV file loading test"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            df = self.converter.load_k_melloddy_data(temp_file)
            self.assertEqual(len(df), 2)
            self.assertIn('Chemical ID', df.columns)
        finally:
            os.unlink(temp_file)
    
    def test_load_k_melloddy_data_xlsx(self):
        """Excel file loading test"""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            self.sample_data.to_excel(f.name, index=False)
            temp_file = f.name
        
        try:
            df = self.converter.load_k_melloddy_data(temp_file)
            self.assertEqual(len(df), 2)
            self.assertIn('Chemical ID', df.columns)
        finally:
            os.unlink(temp_file)
    
    def test_create_conversion_prompt(self):
        """Conversion prompt creation test"""
        sample_record = self.sample_data.iloc[0].to_dict()
        prompt = self.converter.create_conversion_prompt(sample_record)
        
        self.assertIn('K-MELLODDY', prompt)
        self.assertIn('GIST', prompt)
        self.assertIn('Chemical ID', prompt)
    
    @patch('langchain_openai.ChatOpenAI.invoke')
    def test_call_llm_api(self, mock_invoke):
        """LLM API call test"""
        # Mock response setup
        mock_response = Mock()
        mock_response.content = '{"compound_id": "C001"}'
        mock_invoke.return_value = mock_response
        
        prompt = "Test prompt"
        result = self.converter.call_llm_api(prompt)
        
        self.assertEqual(result, '{"compound_id": "C001"}')
        mock_invoke.assert_called_once()
    
    def test_apply_conversion_rules(self):
        """Conversion rules application test"""
        rules = {}
        converted_df = self.converter.apply_conversion_rules(self.sample_data, rules)
        
        # Check column name conversion
        self.assertIn('compound_id', converted_df.columns)
        self.assertIn('compound_name', converted_df.columns)
        self.assertIn('smiles', converted_df.columns)
        self.assertIn('activity_value', converted_df.columns)
        
        # Check if original column names are removed
        self.assertNotIn('Chemical ID', converted_df.columns)
        self.assertNotIn('Chemical Name', converted_df.columns)
    
    def test_apply_default_conversion(self):
        """Default conversion test"""
        converted_df = self.converter.apply_default_conversion(self.sample_data)
        
        self.assertIn('compound_id', converted_df.columns)
        self.assertIn('activity_value', converted_df.columns)
        self.assertEqual(len(converted_df), 2)
    
    def test_save_gist_format(self):
        """GIST format saving test"""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_output.csv')
            
            self.converter.save_gist_format(self.sample_data, output_path)
            
            # Check if file was created
            self.assertTrue(os.path.exists(output_path))
            
            # Check if metadata file was created
            metadata_path = output_path.replace('.csv', '_metadata.json')
            self.assertTrue(os.path.exists(metadata_path))
            
            # Check CSV file content
            saved_df = pd.read_csv(output_path)
            self.assertEqual(len(saved_df), 2)

class TestConversionConfig(unittest.TestCase):
    """Conversion configuration test"""
    
    def test_config_creation(self):
        """Configuration object creation test"""
        config = ConversionConfig(
            api_key="test_key",
            model_name="gemini-1.5-flash",
            source="Gemini"
        )
        
        self.assertEqual(config.api_key, "test_key")
        self.assertEqual(config.model_name, "gemini-1.5-flash")
        self.assertEqual(config.source, "Gemini")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_tokens, 4000)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
