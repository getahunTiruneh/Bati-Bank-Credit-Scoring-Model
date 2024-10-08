import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV


class ModelEvaluator:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.models = {}
        self.results = {}
        self.classification_reports = {}
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        X = self.data.drop(columns=[self.target_column,'CustomerId'])
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("Data split into training and testing sets.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_logistic_regression(self):
        """Train a Logistic Regression model with hyperparameter tuning using GridSearchCV."""
        model = LogisticRegression(max_iter=200)
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'penalty': ['l2', 'none'],
            'solver': ['lbfgs', 'newton-cg', 'saga']
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        self.models['Logistic Regression'] = best_model
        print("Logistic Regression model trained with hyperparameter tuning.")
        print("Best parameters found:", grid_search.best_params_)
        return best_model  # Return the trained model

    def train_random_forest(self):
        """Train a Random Forest model with hyperparameter tuning using GridSearchCV."""
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        self.models['Random Forest'] = best_model
        print("Random Forest model trained with hyperparameter tuning.")
        print("Best parameters found:", grid_search.best_params_)
        return best_model  # Return the trained model

    def plot_roc_curves(self, models, X_test, y_test):
        """
        Plots ROC curves for the given models.
        
        Parameters:
        models (dict): Dictionary containing model names as keys and trained model objects as values.
        X_test (pd.DataFrame or np.ndarray): The test features.
        y_test (pd.Series or np.ndarray): The true test labels.
        """
        # Create subplots for ROC curves
        plt.figure(figsize=(12, 6))

        # Colors for each model's ROC curve
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for i, (model_name, model) in enumerate(models.items()):
            # Predict probabilities for each model
            y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
            
            # Calculate the ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_prob, pos_label=1)  # Ensure pos_label is correct
            roc_auc = roc_auc_score(y_test, y_prob)

            # Subplot for each model's ROC Curve
            plt.subplot(1, len(models), i + 1)
            plt.plot(fpr, tpr, color=colors[i % len(colors)], label=f'ROC curve (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend(loc='lower right')
            plt.grid(True)

        # Show the plots
        plt.tight_layout()
        plt.show()

    def display_classification_reports(self, models, X_test, y_test):
        """Generate classification reports for each model."""
        from sklearn.metrics import classification_report
        
        for model_name, model in models.items():  # Unpack the tuple into model_name and model
            predictions = model.predict(X_test)  # Use the model object to predict
            report = classification_report(y_test, predictions)
            print(f"Classification report generated for {model_name}:\n")  # Use model_name for printing
            print(report)
            print("="*50)
    def plot_model_comparisons(self, models, X_test, y_test):
        """Compare models using Accuracy, Precision, Recall, and F1-Score metrics in percentage."""
        
        # Initialize dictionaries to store metrics for each model
        metrics_dict = {
            'Model': [],
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }
        
        # Loop through each model to calculate and store metrics
        for model_name, model in models.items():
            predictions = model.predict(X_test)
            
            # Generate classification report (as a dictionary)
            report = classification_report(y_test, predictions, output_dict=True)
            
            # Append metrics for each model (multiply by 100 for percentage)
            metrics_dict['Model'].append(model_name)
            metrics_dict['Accuracy'].append(report['accuracy'] * 100)
            metrics_dict['Precision'].append(report['weighted avg']['precision'] * 100)
            metrics_dict['Recall'].append(report['weighted avg']['recall'] * 100)
            metrics_dict['F1-Score'].append(report['weighted avg']['f1-score'] * 100)
        
        # Convert the metrics dictionary to a DataFrame
        metrics_df = pd.DataFrame(metrics_dict)
        
        # Plotting the metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Comparison (%)', fontsize=16)
        
        # Plot for each metric in different subplots
        for i, metric in enumerate(metrics):
            plt.subplot(2, 2, i + 1)
            bar_plot = sns.barplot(x='Model', y=metric, data=metrics_df, palette='viridis')
            plt.title(f'Model Comparison: {metric}')
            plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%
            plt.ylabel(f'{metric} (%)')  # Label with percentage
            plt.grid(True)
            
            # Add percentage annotations on top of the bars
            for p in bar_plot.patches:
                bar_plot.annotate(f'{p.get_height():.2f}%', 
                                (p.get_x() + p.get_width() / 2., 
                                p.get_height()), 
                                ha='center', 
                                va='bottom', 
                                fontsize=10, 
                                color='black', 
                                xytext=(0, 5), 
                                textcoords='offset points')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for the title
        plt.show()
        
    def save_model(self, model, model_name):
        """Saves the provided model object to the specified file path using pickle."""
        # Ensure the directory exists
        directory = 'models'
        os.makedirs(directory, exist_ok=True)
        
        # Construct the file path
        file_path = os.path.join(directory, f"{model_name}.pkl")
        
        # Save the model
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"{model_name} model saved to {file_path}")