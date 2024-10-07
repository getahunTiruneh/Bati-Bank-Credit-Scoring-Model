import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

class ModelEvaluator:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.models = {}
        self.results = {}
        self.classification_reports = {}
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Data split into training and testing sets.")

    def train_logistic_regression(self):
        """Train a Logistic Regression model."""
        model = LogisticRegression(max_iter=200)  # Increased max_iter for convergence
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        print("Logistic Regression model trained.")

    def train_random_forest(self):
        """Train a Random Forest model."""
        model = RandomForestClassifier()
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        print("Random Forest model trained.")

    def predict_probabilities(self):
        """Predict probabilities using the trained models."""
        for name, model in self.models.items():
            self.results[name] = model.predict_proba(self.X_test)[:, 1]  # Probabilities for the positive class
            print(f"Predicted probabilities for {name}.")

    def calculate_roc_curves(self):
        """Calculate ROC curves for the models."""
        self.roc_curves = {}
        for name, probs in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, probs, pos_label=1)  # Ensure pos_label is correct
            roc_auc = roc_auc_score(self.y_test, probs)
            self.roc_curves[name] = (fpr, tpr, roc_auc)
            print(f"Calculated ROC curve for {name} (AUC = {roc_auc:.4f}).")

    def plot_roc_curves(self):
        """Plot the ROC curves."""
        plt.figure(figsize=(12, 6))
        
        for name, (fpr, tpr, roc_auc) in self.roc_curves.items():
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        print("ROC curves plotted.")

    def generate_classification_reports(self):
        """Generate classification reports for each model."""
        for name, model in self.models.items():
            predictions = model.predict(self.X_test)
            self.classification_reports[name] = classification_report(self.y_test, predictions)
            print(f"Classification report generated for {name}.")

    def display_classification_reports(self):
        """Display classification reports."""
        for name, report in self.classification_reports.items():
            print(f"Classification Report for {name}:\n")
            print(report)
            print("="*50)