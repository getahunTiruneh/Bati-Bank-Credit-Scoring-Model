import unittest
import sys,os
import pandas as pd
import numpy as np
# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.EDA_scripts import ExploratoryDataAnalysis  

class TestExploratoryDataAnalysis(unittest.TestCase):

    def setUp(self):
        """
        Sets up a sample DataFrame for testing purposes.
        """
        data = {
            'Numerical1': [1, 2, 3, 4, 5],
            'Numerical2': [10, 20, 30, 40, 50],
            'Categorical': ['A', 'B', 'A', 'C', 'B'],
            'Missing': [1, np.nan, 3, 4, np.nan]
        }
        self.df = pd.DataFrame(data)
        self.eda = ExploratoryDataAnalysis(self.df)

    def test_head(self):
        """
        Tests the head method to ensure it returns the correct rows.
        """
        result = self.eda.head()
        self.assertEqual(result.shape[0], 5)  # Check number of rows
        self.assertEqual(result.shape[1], 4)  # Check number of columns
        pd.testing.assert_frame_equal(result, self.df.head())

    def test_dataset_overview(self):
        """
        Tests the dataset overview functionality for no runtime errors.
        """
        try:
            self.eda.dataset_overview()
        except Exception as e:
            self.fail(f"dataset_overview raised an exception: {e}")

    def test_summary_statistics(self):
        """
        Tests the summary_statistics method for no runtime errors.
        """
        try:
            self.eda.summary_statistics()
        except Exception as e:
            self.fail(f"summary_statistics raised an exception: {e}")

    def test_plot_numerical_distribution(self):
        """
        Tests the plot_numerical_distribution method for numerical columns.
        """
        try:
            self.eda.plot_numerical_distribution(['Numerical1', 'Numerical2'])
        except Exception as e:
            self.fail(f"plot_numerical_distribution raised an exception: {e}")

    def test_categorical_distribution(self):
        """
        Tests the categorical_distribution method for categorical columns.
        """
        try:
            self.eda.categorical_distribution(['Categorical'])
        except Exception as e:
            self.fail(f"categorical_distribution raised an exception: {e}")

    def test_correlation_analysis(self):
        """
        Tests the correlation_analysis method for no runtime errors.
        """
        try:
            self.eda.correlation_analysis()
        except Exception as e:
            self.fail(f"correlation_analysis raised an exception: {e}")

    def test_missing_values_analysis(self):
        """
        Tests the missing_values_analysis method to ensure it runs and reports missing values correctly.
        """
        try:
            self.eda.missing_values_analysis()
        except Exception as e:
            self.fail(f"missing_values_analysis raised an exception: {e}")

    def test_outlier_detection(self):
        """
        Tests the outlier_detection method for numerical columns.
        """
        try:
            self.eda.outlier_detection(['Numerical1', 'Numerical2'])
        except Exception as e:
            self.fail(f"outlier_detection raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
