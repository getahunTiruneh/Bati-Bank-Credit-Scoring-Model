import unittest
import sys, os
import pandas as pd
import numpy as np
from datetime import datetime
# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.feature_engineering_scripts import FeatureEngineering  # Import the class to be tested


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {
            'CustomerId': [1, 1, 2, 2, 3],
            'Amount': [100, 200, 150, np.nan, 300],
            'TransactionId': [101, 102, 201, 202, 301],
            'TransactionStartTime': pd.to_datetime([
                '2024-11-01 08:00:00', 
                '2024-11-01 09:00:00',
                '2024-11-02 10:00:00',
                '2024-11-02 11:00:00',
                '2024-11-03 12:00:00'
            ]),
            'ProviderId': ['A', 'B', 'A', 'B', 'C'],
            'ProductId': ['P1', 'P2', 'P1', 'P3', 'P2'],
            'ProductCategory': ['Cat1', 'Cat2', 'Cat1', 'Cat3', 'Cat2'],
            'ChannelId': ['Online', 'Offline', 'Online', 'Offline', 'Online']
        }
        self.df = pd.DataFrame(data)
        self.feature_engineering = FeatureEngineering(self.df)

    def test_remove_high_missing_values(self):
        # Add a column with high missing values
        self.df['HighMissing'] = [np.nan, np.nan, np.nan, 4, 5]
        processed_df = self.feature_engineering.remove_high_missing_values(threshold=0.5)
        self.assertNotIn('HighMissing', processed_df.columns)
    
    def test_create_aggregate_features(self):
        processed_df = self.feature_engineering.create_aggregate_features()
        self.assertIn('Total_Transaction_Amount', processed_df.columns)
        self.assertEqual(processed_df['Total_Transaction_Amount'][0], 300)  # Sum of amounts for CustomerId 1
    
    def test_extract_temporal_features(self):
        processed_df = self.feature_engineering.extract_temporal_features()
        self.assertIn('Transaction_Hour', processed_df.columns)
        self.assertEqual(processed_df['Transaction_Hour'][0], 8)  # First transaction hour
    
    def test_handle_missing_values(self):
        processed_df = self.feature_engineering.handle_missing_values(method='imputation', strategy='mean')
        self.assertFalse(processed_df['Amount'].isnull().any())
        self.assertAlmostEqual(processed_df['Amount'][3], self.df['Amount'].mean(skipna=True))
    
    def test_encode_categorical_variables(self):
        processed_df = self.feature_engineering.encode_categorical_variables(encoding_type='onehot')
        self.assertTrue(any(col.startswith('ProviderId') for col in processed_df.columns))
        self.assertTrue(any(col.startswith('ChannelId') for col in processed_df.columns))


if __name__ == '__main__':
    unittest.main()