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
import logging
from datetime import datetime
import mlflow
import mlflow.sklearn


class ModelEvaluator:
    def __init__(self, data, target_column, experiment_name="Model_Evaluation"):
        self.data = data
        self.target_column = target_column
        self.models = {}
        self.results = {}
        self.classification_reports = {}

        # Setup logging
        self.setup_logging()
        logging.info("ModelEvaluator initialized.")
        # Setup MLflow
        mlflow.set_experiment(experiment_name)
        logging.info(f"MLflow experiment '{experiment_name}' initialized.")

    def setup_logging(self):
        """Set up logging to track operations and save logs to a file."""
        log_dir = "../logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"model_evaluator_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        logging.info("Logging setup complete. Logs will be saved to: %s", log_file)

    def split_data(self, test_size=0.2, random_state=42):
        """Split the data into training and testing sets."""
        X = self.data.drop(columns=[self.target_column, 'CustomerId'])
        y = self.data[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logging.info("Data split into training and testing sets.")
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

        with mlflow.start_run(run_name="Logistic Regression"):
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("Train Score", grid_search.best_score_)
            mlflow.sklearn.log_model(best_model, "logistic_regression_model")
            logging.info("Logistic Regression model logged to MLflow.")
        
        self.models['Logistic Regression'] = best_model
        return best_model

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

        with mlflow.start_run(run_name="Random Forest"):
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("Train Score", grid_search.best_score_)
            mlflow.sklearn.log_model(best_model, "random_forest_model")
            logging.info("Random Forest model logged to MLflow.")
        
        self.models['Random Forest'] = best_model
        return best_model

    def save_model(self, model, model_name):
        """Saves the provided model object to the specified file path using pickle."""
        try:
            directory = 'models'
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, f"{model_name}.pkl")
            with open(file_path, 'wb') as file:
                pickle.dump(model, file)
            logging.info("%s model saved to %s", model_name, file_path)
        except Exception as e:
            logging.error("Error saving model %s: %s", model_name, str(e))
            raise e

    def plot_roc_curves(self):
        """Plot ROC curves for all trained models."""
        try:
            plt.figure(figsize=(10, 8))
            for model_name, model in self.models.items():
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                auc = roc_auc_score(self.y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.2f})')
            plt.title('ROC Curves')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.grid()
            plt.show()
            logging.info("ROC curves plotted successfully.")
        except Exception as e:
            logging.error("Error plotting ROC curves: %s", str(e))
            raise e

    def display_classification_reports(self):
        """Display classification reports for all trained models."""
        try:
            for model_name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                report = classification_report(self.y_test, y_pred)
                self.classification_reports[model_name] = report
                logging.info("Classification report for %s:\n%s", model_name, report)
                print(f"\n{model_name} Classification Report:\n{report}")
        except Exception as e:
            logging.error("Error displaying classification reports: %s", str(e))
            raise e

    def plot_model_comparisons(self):
        """Compare model performance using bar plots of AUC scores."""
        try:
            auc_scores = {}
            for model_name, model in self.models.items():
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                auc_scores[model_name] = roc_auc_score(self.y_test, y_pred_proba)

            plt.figure(figsize=(8, 6))
            sns.barplot(x=list(auc_scores.keys()), y=list(auc_scores.values()))
            plt.title('Model Comparison (AUC Scores)')
            plt.ylabel('AUC Score')
            plt.xlabel('Model')
            plt.grid(axis='y')
            plt.show()
            logging.info("Model comparisons plotted successfully.")
        except Exception as e:
            logging.error("Error plotting model comparisons: %s", str(e))
            raise e

    def plot_metric_comparisons(self):
        """
        Compare model performance using multiple metrics like Accuracy, Precision, Recall, and F1-Score.
        Visualize the metrics for each model using a bar plot with a consistent and aesthetic color palette.
        """
        try:
            # Dictionary to store metrics for each model
            metrics_data = {
                "Model": [],
                "Accuracy": [],
                "Precision": [],
                "Recall": [],
                "F1-Score": []
            }

            # Calculate metrics for each model
            for model_name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                report = classification_report(self.y_test, y_pred, output_dict=True)

                metrics_data["Model"].append(model_name)
                metrics_data["Accuracy"].append(report["accuracy"])
                metrics_data["Precision"].append(report["weighted avg"]["precision"])
                metrics_data["Recall"].append(report["weighted avg"]["recall"])
                metrics_data["F1-Score"].append(report["weighted avg"]["f1-score"])

            # Convert the metrics dictionary to a DataFrame for visualization
            metrics_df = pd.DataFrame(metrics_data)

            # Melt the DataFrame for seaborn compatibility
            metrics_melted = metrics_df.melt(id_vars="Model", var_name="Metric", value_name="Score")

            # Plot the metrics using a bar plot
            plt.figure(figsize=(12, 8))
            sns.barplot(data=metrics_melted, x="Metric", y="Score", hue="Model", palette="coolwarm")
            plt.title("Model Performance Comparison Across Metrics", fontsize=16)
            plt.xlabel("Metric", fontsize=14)
            plt.ylabel("Score", fontsize=14)
            plt.legend(title="Model", fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            logging.info("Model metric comparisons plotted successfully.")
        except Exception as e:
            logging.error("Error plotting model metric comparisons: %s", str(e))
            raise e
