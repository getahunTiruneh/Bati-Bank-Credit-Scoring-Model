import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Ensure the log directory exists
log_dir = "../log"
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, "eda_analysis.log"), mode='w')
    ]
)

class ExploratoryDataAnalysis:
    def __init__(self, df):
        """
        Initializes the ExploratoryDataAnalysis class with a DataFrame.
        """
        self.df = df
        logging.info("ExploratoryDataAnalysis initialized with DataFrame of shape %s", df.shape)

    def head(self):
        """
        Returns the first five rows of the DataFrame.
        """
        logging.info("Fetching the first five rows of the DataFrame.")
        return self.df.head()

    def dataset_overview(self):
        """
        Displays an overview of the dataset, including the number of rows,
        number of columns, data types, and the first five rows.
        """
        logging.info("Displaying dataset overview.")
        print("Dataset Overview:")
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")
        print("\nColumn Data Types:")
        print(self.df.dtypes)
        print("\nFirst 5 rows of the dataset:")
        print(self.df.head())

    def summary_statistics(self):
        """
        Displays summary statistics for numerical columns in the dataset.
        """
        logging.info("Calculating summary statistics.")
        print("\nSummary Statistics:")
        print(self.df.describe())

    def plot_numerical_distribution(self, numerical_cols):
        """
        Visualizes the distribution of numerical features using histograms with and without 
        outlier handling and applying log transformation.
        """
        logging.info("Plotting numerical distributions for columns: %s", numerical_cols)
        for col in numerical_cols:
            try:
                plt.figure(figsize=(12, 5))
                
                # Original Data Distribution
                plt.subplot(1, 2, 1)
                sns.histplot(self.df[col], bins=30, kde=True)
                plt.title(f'Distribution of {col} (Original)')
                plt.xlabel(col)
                plt.ylabel('Frequency')
                plt.grid(True)

                # Log Transformed Data Distribution
                log_transformed_data = np.log1p(self.df[col])
                plt.subplot(1, 2, 2)
                sns.histplot(log_transformed_data, bins=30, kde=True)
                plt.title(f'Distribution of {col} (Log Scale)')
                plt.xlabel(f'Log of {col}')
                plt.ylabel('Frequency')
                plt.grid(True)

                plt.tight_layout()
                plt.show()
            except Exception as e:
                logging.error("Error plotting numerical distribution for column %s: %s", col, e)

    def categorical_distribution(self, categorical_cols):
        """
        Visualizes the distribution of categorical features using count plots.
        """
        logging.info("Plotting categorical distributions for columns: %s", categorical_cols)
        for col in categorical_cols:
            try:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=self.df, x=col, palette='Set2')
                plt.title(f'Distribution of {col}')
                plt.xticks(rotation=45)
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.show()
            except Exception as e:
                logging.error("Error plotting categorical distribution for column %s: %s", col, e)

    def correlation_analysis(self):
        """
        Performs correlation analysis on numerical features and visualizes
        the correlation matrix using a heatmap.
        """
        logging.info("Performing correlation analysis.")
        try:
            numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
            corr_matrix = self.df[numerical_cols].corr()
            plt.figure(figsize=(12, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.title('Correlation Matrix of Numerical Features')
            plt.show()
        except Exception as e:
            logging.error("Error performing correlation analysis: %s", e)

    def missing_values_analysis(self):
        """
        Identifies and visualizes missing values in the dataset.
        """
        logging.info("Analyzing missing values.")
        missing_values = self.df.isnull().sum()
        print("\nMissing Values Analysis:")
        print(missing_values)
        logging.info("Missing values per column: %s", missing_values.to_dict())

    def outlier_detection(self, numerical_cols):
        """
        Detects and visualizes outliers in numerical columns using box plots.
        """
        logging.info("Detecting outliers for columns: %s", numerical_cols)
        for col in numerical_cols:
            try:
                plt.figure(figsize=(10, 6))
                sns.boxplot(data=self.df, y=col)
                plt.title(f'Boxplot of {col}')
                plt.ylabel(col)
                plt.show()
            except Exception as e:
                logging.error("Error detecting outliers for column %s: %s", col, e)
