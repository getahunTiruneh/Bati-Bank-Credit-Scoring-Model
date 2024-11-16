import unittest
import sys,os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# Add the project root to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from scripts.model_development_scripts import ModelEvaluator  

class TestModelEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a mock dataset to use for testing."""
        # Create a mock binary classification dataset
        X, y = make_classification(
            n_samples=500, 
            n_features=10, 
            n_informative=5, 
            n_redundant=2, 
            random_state=42
        )
        cls.mock_data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        cls.mock_data['CustomerId'] = range(1, 501)  # Add a CustomerId column
        cls.mock_data['target'] = y

    def setUp(self):
        """Set up ModelEvaluator instance for each test."""
        self.evaluator = ModelEvaluator(self.mock_data, target_column='target')
        self.evaluator.split_data()

    def test_split_data(self):
        """Test data splitting functionality."""
        X_train, X_test, y_train, y_test = self.evaluator.split_data()
        self.assertEqual(len(X_train), 400)
        self.assertEqual(len(X_test), 100)
        self.assertEqual(len(y_train), 400)
        self.assertEqual(len(y_test), 100)

    def test_train_logistic_regression(self):
        """Test Logistic Regression training and hyperparameter tuning."""
        model = self.evaluator.train_logistic_regression()
        self.assertIsInstance(model, LogisticRegression)
        self.assertIn('Logistic Regression', self.evaluator.models)

    def test_train_random_forest(self):
        """Test Random Forest training and hyperparameter tuning."""
        model = self.evaluator.train_random_forest()
        self.assertIsInstance(model, RandomForestClassifier)
        self.assertIn('Random Forest', self.evaluator.models)

    def test_plot_roc_curves(self):
        """Test ROC curve plotting."""
        # Train models for testing
        self.evaluator.train_logistic_regression()
        self.evaluator.train_random_forest()
        
        # Check that no exceptions are raised during plotting
        try:
            self.evaluator.plot_roc_curves(self.evaluator.models, self.evaluator.X_test, self.evaluator.y_test)
        except Exception as e:
            self.fail(f"plot_roc_curves raised an exception: {e}")

    def test_display_classification_reports(self):
        """Test classification report generation."""
        # Train models for testing
        self.evaluator.train_logistic_regression()
        self.evaluator.train_random_forest()
        
        # Check that no exceptions are raised during report display
        try:
            self.evaluator.display_classification_reports(self.evaluator.models, self.evaluator.X_test, self.evaluator.y_test)
        except Exception as e:
            self.fail(f"display_classification_reports raised an exception: {e}")

    def test_plot_model_comparisons(self):
        """Test model comparison plotting."""
        # Train models for testing
        self.evaluator.train_logistic_regression()
        self.evaluator.train_random_forest()
        
        # Check that no exceptions are raised during comparison plotting
        try:
            self.evaluator.plot_model_comparisons(self.evaluator.models, self.evaluator.X_test, self.evaluator.y_test)
        except Exception as e:
            self.fail(f"plot_model_comparisons raised an exception: {e}")

    def test_save_model(self):
        """Test model saving functionality."""
        # Train a Logistic Regression model
        model = self.evaluator.train_logistic_regression()
        
        # Save the model
        model_name = 'Test_Logistic_Regression'
        self.evaluator.save_model(model, model_name)
        
        # Check if the file was created
        import os
        self.assertTrue(os.path.exists(f'models/{model_name}.pkl'))

if __name__ == '__main__':
    unittest.main()
