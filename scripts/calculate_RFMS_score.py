import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scorecardpy as sc
import seaborn as sns
import os

class RFMSRiskClassifier:
    def __init__(self, df):
        """
        Initialize the RFMSRiskClassifier with a DataFrame.
        
        Parameters:
        - df: DataFrame containing transaction data with customer information.
        """
        self.df = df

    def calculate_recency(self, current_date):
        """
        Calculate the recency (days since last transaction) for each customer.

        Parameters:
        - current_date: The current date used to calculate recency.

        Returns:
        - DataFrame with a new column 'Recency' representing the recency for each customer.
        """
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'], errors='coerce').dt.tz_localize(None)
        current_date = pd.to_datetime(current_date).tz_localize(None)
        self.df['Recency'] = (current_date - self.df['TransactionStartTime']).dt.days
        
        return self.df

    def calculate_frequency(self):
        """
        Calculate the frequency (transaction count) for each customer.

        Returns:
        - DataFrame with a new column 'Frequency' representing the transaction count for each customer.
        """
        self.df['Frequency'] = self.df['Transaction_Count']
        return self.df

    def calculate_monetary(self):
        """
        Calculate the monetary value (total transaction amount) for each customer.

        Returns:
        - DataFrame with a new column 'Monetary' representing the total transaction amount for each customer.
        """
        self.df['Monetary'] = self.df.groupby('CustomerId')['Total_Transaction_Amount'].transform('sum')
        return self.df

    def calculate_seasonality(self):
        """
        Calculate the seasonality (number of unique transaction months) for each customer.

        Returns:
        - DataFrame with a new column 'Seasonality' representing the unique months of transactions for each customer.
        """
        self.df['Transaction_Month'].fillna(0, inplace=True)
        self.df['Seasonality'] = self.df.groupby('CustomerId')['Transaction_Month'].transform(lambda x: x.nunique())
        return self.df

    def normalize_rfms(self):
        """
        Normalize the RFMS (Recency, Frequency, Monetary, Seasonality) columns to a 0-1 scale.

        Returns:
        - DataFrame with normalized RFMS columns.
        """
        rfms_columns = ['Recency', 'Frequency', 'Monetary', 'Seasonality']
        self.df[rfms_columns] = self.df[rfms_columns].apply(pd.to_numeric, errors='coerce')
        self.df[rfms_columns] = self.df[rfms_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return self.df

    def assign_risk_category(self, threshold=0.25):
        """
        Assign risk categories based on RFMS scores.

        Parameters:
        - threshold: A threshold value to classify customers as 'good' or 'bad'.

        Returns:
        - DataFrame with a new column 'Risk_category' containing the risk classification.
        """
        self.df['RFMS_score'] = self.df[['Recency', 'Frequency', 'Monetary', 'Seasonality']].mean(axis=1)
        self.df['Risk_category'] = self.df['RFMS_score'].apply(lambda x: 'good' if x >= threshold else 'bad')
        return self.df
    def calculate_woe_iv(df, feature, target):
        """
        Calculate Weight of Evidence (WoE) and Information Value (IV) for a given feature.

        Parameters:
        - df: DataFrame containing the data.
        - feature: Column name for which to calculate WoE and IV.
        - target: Binary column name representing the target variable (1 for good, 0 for bad).

        Returns:
        - DataFrame with feature names and their corresponding IV values.
        """
        results = []
        total_events = df[target].sum()
        total_non_events = df[target].count() - total_events
        
        grouped = df.groupby(feature)[target].agg(['count', 'sum']).reset_index()
        grouped.rename(columns={'count': 'total', 'sum': 'events'}, inplace=True)
        
        grouped['non_events'] = grouped['total'] - grouped['events']
        grouped.replace({'events': {0: np.nan}, 'non_events': {0: np.nan}}, inplace=True)
        
        grouped['woe'] = np.log((grouped['events'] / total_events) / (grouped['non_events'] / total_non_events))
        grouped['iv'] = (grouped['events'] / total_events - grouped['non_events'] / total_non_events) * grouped['woe']
        
        total_iv = grouped['iv'].sum()
        info_values = pd.DataFrame({
            'Feature': [feature],
            'IV': [total_iv]
        })
        
        return info_values
    def save_merged_data(self, final_merged_df, output_file, file_path):
        """
        Save the final merged DataFrame as a CSV file.
        Parameters:
            - final_merged_df: The DataFrame to be saved.
            - output_file: The name of the CSV file.
            - file_path: The directory path to save the file.
        Returns:
        - Saves the DataFrame to the specified path as a CSV file.
        """
        os.makedirs(file_path, exist_ok=True)  
        final_merged_df.to_csv(f"{file_path}/{output_file}.csv", index=False)
        print(f"Merged data saved to {file_path}/{output_file}.csv")