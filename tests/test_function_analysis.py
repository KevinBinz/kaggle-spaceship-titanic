import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from feature_engineer import FeatureEngineer

class TestFunctionAnalysis(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.fe = FeatureEngineer()

    def test_preprocess_raw_data_calls(self):
        """Test that preprocess_raw_data calls the expected functions."""
        preprocessing_functions, _ = self.fe.get_used_functions()
        expected_calls = [
            '_calculate_spending_features',
            '_extract_cabin_components',
            '_extract_group_components',
            '_extract_name_components'
        ]
        self.assertEqual(set(preprocessing_functions), set(expected_calls))

    def test_engineer_features_calls(self):
        """Test that engineer_features calls the expected functions."""
        _, engineering_functions = self.fe.get_used_functions()
        expected_calls = [
            'preprocess_raw_data',
            'create_age_features',
            'create_cabin_features',
            'create_family_features',
            'create_transport_features',
            'impute_missing_values'
        ]
        self.assertEqual(set(engineering_functions), set(expected_calls))

    def test_no_function_calls(self):
        """Test that __init__ has no function calls."""
        calls = self.fe._analyze_function_calls(self.fe.__init__)
        self.assertEqual(calls, [])

if __name__ == '__main__':
    unittest.main(verbosity=2) 