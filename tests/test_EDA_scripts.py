import unittest
import pandas as pd
import sys
import os

# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.EDA_scripts import ExploratoryDataAnalysis
from unittest.mock import patch
import io

class TestExploratoryDataAnalysis(unittest.TestCase):
    def setUp(self):
        # Create a simple DataFrame for testing
        self.df = pd.DataFrame({
            'Amount': [1, 2, 3],
            'Value': [4, 5, 6],
            'PricingStrategy': [1, 0, 1],
            'FraudResult': [0, 1, 0],
            'CountryCode': ['US', 'UK', 'DE']
        })
        self.eda = ExploratoryDataAnalysis(self.df)

    @patch('sys.stdout', new_callable=io.StringIO)  # Patch stdout
    def test_dataset_overview(self, mock_stdout):
        self.eda.dataset_overview()  # Call the method
        output = mock_stdout.getvalue()  # Get printed output
        self.assertIn("Dataset Overview:", output)
        self.assertIn("Number of rows: 3", output)
        self.assertIn("Number of columns: 5", output)

    @patch('sys.stdout', new_callable=io.StringIO)  # Patch stdout
    def test_summary_statistics(self, mock_stdout):
        self.eda.summary_statistics()  # Call the method
        output = mock_stdout.getvalue()  # Get printed output
        self.assertIn("Summary Statistics:", output)
        self.assertIn("Amount", output)  # Check for presence of a column in stats
        self.assertIn("Value", output)  # Check for presence of a column in stats

    # You can add more tests for other methods as needed...

if __name__ == '__main__':
    unittest.main()
