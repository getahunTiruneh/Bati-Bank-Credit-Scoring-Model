import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class ExploratoryDataAnalysis:
    def __init__(self, df):
        """
        Initializes the ExploratoryDataAnalysis class with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to be analyzed.
        """
        self.df = df

    def head(self):
        """
        Returns the first five rows of the DataFrame.

        Returns:
        pd.DataFrame: The top five rows of the dataset.
        """
        return self.df.head()

    def dataset_overview(self):
        """
        Displays an overview of the dataset, including the number of rows,
        number of columns, data types, and the first five rows.
        """
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
        print("\nSummary Statistics:")
        print(self.df.describe())

    def plot_numerical_distribution(self, numerical_cols):
        """
        Visualizes the distribution of numerical features using histograms with and without 
        outlier handling and applying log transformation.

        Parameters:
        df (DataFrame): The DataFrame containing the data.
        numerical_cols (list): A list of numerical column names to visualize.
        """
        
        for col in numerical_cols:
            # 1. Plotting the original distribution without handling outliers
            plt.figure(figsize=(12, 5))

            # Original Data Distribution
            plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
            sns.histplot(self.df[col], bins=30, kde=True)
            plt.title(f'Distribution of {col} (Original)')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True)

            # 2. Applying log transformation
            log_transformed_data = np.log1p(self.df[col])  # Using log(1 + x)

            # Log Transformed Data Distribution
            plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
            sns.histplot(log_transformed_data, bins=30, kde=True)
            plt.title(f'Distribution of {col} (Log Scale)')
            plt.xlabel(f'Log of {col}')
            plt.ylabel('Frequency')
            plt.grid(True)

            plt.tight_layout()  # Adjusts the layout to prevent overlap
            plt.show()

    def categorical_distribution(self, categorical_cols):
        """
        Visualizes the distribution of categorical features using count plots.

        Parameters:
        categorical_cols (list): A list of categorical column names to visualize.
        """
        print("\nDistribution of Categorical Features:")
        for col in categorical_cols:
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.df, x=col,palette='Set2')  # Count plot for categorical data
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
            plt.xlabel(col)  # Labeling the x-axis
            plt.ylabel('Count')  # Labeling the y-axis
            plt.show()

    def correlation_analysis(self):
        """
        Performs correlation analysis on numerical features and visualizes
        the correlation matrix using a heatmap.
        """
        print("\nCorrelation Analysis:")
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.df[numerical_cols].corr()  # Calculate correlation matrix
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')  # Heatmap with annotations
        plt.title('Correlation Matrix of Numerical Features')
        plt.show()

    def missing_values_analysis(self):
        """
        Identifies and visualizes missing values in the dataset.
        """
        print("\nMissing Values Analysis:")
        print(self.df.isnull().sum())  # Count missing values per column

    def outlier_detection(self, numerical_cols):
        """
        Detects and visualizes outliers in numerical columns using box plots.

        Parameters:
        numerical_cols (list): A list of numerical column names to visualize.
        """
        print("\nOutlier Detection:")
        for col in numerical_cols:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.df, y=col)  # Box plot for outlier detection
            plt.title(f'Boxplot of {col}')
            plt.ylabel(col)  # Labeling the y-axis
            plt.show()
