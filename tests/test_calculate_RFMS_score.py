import unittest
import sys, os
import pandas as pd
import numpy as np
from datetime import datetime
# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.calculate_RFMS_score import RFMSRiskClassifier  

class TestRFMSRiskClassifier(unittest.TestCase):

    def setUp(self):
        """Set up a sample DataFrame to be used for testing."""
        data = {
            'CustomerId': [1, 2, 1, 3, 2],
            'TransactionStartTime': [
                '2024-01-10', '2024-01-11', '2024-02-10', '2024-01-09', '2024-02-10'
            ],
            'Transaction_Count': [5, 3, 2, 4, 1],
            'Total_Transaction_Amount': [100, 200, 150, 300, 250],
            'Transaction_Month': [1, 1, 2, 1, 2]
        }
        self.df = pd.DataFrame(data)
        self.classifier = RFMSRiskClassifier(self.df)

    def test_calculate_recency(self):
        """Test if recency is calculated correctly."""
        current_date = '2024-02-20'
        df_result = self.classifier.calculate_recency(current_date)
        expected_recency = [41, 40, 10, 42, 10]
        self.assertListEqual(df_result['Recency'].tolist(), expected_recency)

    def test_calculate_frequency(self):
        """Test if frequency is set correctly from the Transaction_Count column."""
        df_result = self.classifier.calculate_frequency()
        expected_frequency = [5, 3, 2, 4, 1]
        self.assertListEqual(df_result['Frequency'].tolist(), expected_frequency)

    def test_calculate_monetary(self):
        """Test if monetary is calculated as the sum of transactions for each customer."""
        df_result = self.classifier.calculate_monetary()
        expected_monetary = [250, 450, 250, 300, 450]
        self.assertListEqual(df_result['Monetary'].tolist(), expected_monetary)

    def test_calculate_seasonality(self):
        """Test if seasonality is calculated as the number of unique months for each customer."""
        df_result = self.classifier.calculate_seasonality()
        expected_seasonality = [2, 2, 2, 1, 2]
        self.assertListEqual(df_result['Seasonality'].tolist(), expected_seasonality)

    def test_normalize_rfms(self):
        """Test if the RFMS columns are normalized to the range 0-1."""
        # Set up data with known min/max values for easier testing
        self.classifier.calculate_recency('2024-02-20')
        self.classifier.calculate_frequency()
        self.classifier.calculate_monetary()
        self.classifier.calculate_seasonality()
        df_result = self.classifier.normalize_rfms()
        self.assertTrue((df_result[['Recency', 'Frequency', 'Monetary', 'Seasonality']].max() <= 1).all())
        self.assertTrue((df_result[['Recency', 'Frequency', 'Monetary', 'Seasonality']].min() >= 0).all())

    def test_assign_risk_category(self):
        """Test if risk categories are assigned correctly based on a threshold."""
        self.classifier.calculate_recency('2024-02-20')
        self.classifier.calculate_frequency()
        self.classifier.calculate_monetary()
        self.classifier.calculate_seasonality()
        self.classifier.normalize_rfms()
        df_result = self.classifier.assign_risk_category(threshold=0.25)
        # Example expected output; will vary based on calculated RFMS scores
        # Check if 'Risk_category' is either 'good' or 'bad'
        self.assertTrue(set(df_result['Risk_category'].unique()).issubset({'good', 'bad'}))

    def test_calculate_woe_iv(self):
        """Test if WoE and IV are calculated correctly for a binary feature."""
        # Add a binary target column for testing
        self.df['target'] = [1, 0, 1, 0, 1]
        iv_result = RFMSRiskClassifier.calculate_woe_iv(self.df, 'Transaction_Month', 'target')
        self.assertIn('IV', iv_result.columns)
        self.assertGreaterEqual(iv_result['IV'].iloc[0], 0)

    def test_save_merged_data(self):
        """Test if data is saved to the specified file path."""
        output_file = 'test_output'
        file_path = './temp'
        self.classifier.save_merged_data(self.df, output_file, file_path)
        file_exists = os.path.exists(f"{file_path}/{output_file}.csv")
        self.assertTrue(file_exists)
        # Clean up
        os.remove(f"{file_path}/{output_file}.csv")
        os.rmdir(file_path)

if __name__ == '__main__':
    unittest.main()
