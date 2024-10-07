import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

class RFMSRiskClassifier:
    def __init__(self, df):
        self.df = df

    def calculate_recency(self, current_date):
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'], errors='coerce').dt.tz_localize(None)
        current_date = pd.to_datetime(current_date).tz_localize(None)
        self.df['Recency'] = (current_date - self.df['TransactionStartTime']).dt.days
        
        return self.df

    def calculate_frequency(self):
        self.df['Frequency'] = self.df['Transaction_Count']
        return self.df

    def calculate_monetary(self):
        self.df['Monetary'] = self.df.groupby('CustomerId')['Total_Transaction_Amount'].transform('sum')
        return self.df

    def calculate_seasonality(self):
        # Ensure no missing values in Transaction_Month, or handle them appropriately
        self.df['Transaction_Month'].fillna(0, inplace=True)

        # Calculate Seasonality (unique months of transactions per CustomerId)
        self.df['Seasonality'] = self.df.groupby('CustomerId')['Transaction_Month'].transform(lambda x: x.nunique())
        return self.df

    def normalize_rfms(self):
        rfms_columns = ['Recency', 'Frequency', 'Monetary', 'Seasonality']
        # Ensure RFMS columns are numeric
        self.df[rfms_columns] = self.df[rfms_columns].apply(pd.to_numeric, errors='coerce')
        self.df[rfms_columns] = self.df[rfms_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return self.df

    def assign_risk_category(self, threshold=0.25):
        self.df['RFMS_score'] = self.df[['Recency', 'Frequency', 'Monetary', 'Seasonality']].mean(axis=1)
        self.df['Risk_category'] = self.df['RFMS_score'].apply(lambda x: 'good' if x >= threshold else 'bad')
        return self.df

    @staticmethod
    def calculate_woe_iv(df, feature, target):
        """
        Calculate Weight of Evidence (WoE) and Information Value (IV) for a given feature.
        
        Parameters:
        - df: DataFrame containing the data
        - feature: Column name for which to calculate WoE and IV
        - target: Binary column name representing the target variable (1 for good, 0 for bad)
        
        Returns:
        - DataFrame with variable names and their corresponding IV values.
        """
        # Create a DataFrame to hold the results
        results = []

        # Calculate the total number of events and non-events
        total_events = df[target].sum()
        total_non_events = df[target].count() - total_events
        
        # Group by the feature and calculate the count of events and non-events
        grouped = df.groupby(feature)[target].agg(['count', 'sum']).reset_index()
        grouped.rename(columns={'count': 'total', 'sum': 'events'}, inplace=True)
        
        # Calculate non-events
        grouped['non_events'] = grouped['total'] - grouped['events']
        
        # Avoid division by zero
        grouped.replace({'events': {0: np.nan}, 'non_events': {0: np.nan}}, inplace=True)
        
        # Calculate WoE and IV
        grouped['woe'] = np.log((grouped['events'] / total_events) / (grouped['non_events'] / total_non_events))
        grouped['iv'] = (grouped['events'] / total_events - grouped['non_events'] / total_non_events) * grouped['woe']
        
        # Aggregate IV values
        total_iv = grouped['iv'].sum()
        
        info_values = pd.DataFrame({
            'Feature': [feature],  # Include the feature name
            'IV': [total_iv]       # Replace with actual IV calculation
        })
        
        return info_values

    @staticmethod
    def plot_woe_binned(df, feature, target, bins=10):
        """
        Create a WoE plot for a specified feature with binning.
        
        Parameters:
        - df: DataFrame containing the data
        - feature: Column name for which to create the WoE plot
        - target: Binary column name representing the target variable (1 for good, 0 for bad)
        - bins: Number of bins for continuous variables
        """
        # Use qcut for better distribution of bins
        df['binned_feature'] = pd.qcut(df[feature], q=bins, duplicates='drop')

        # Create a DataFrame to hold the WoE values and counts
        results = []

        # Calculate total events and non-events
        total_events = df[target].sum()
        total_non_events = df[target].count() - total_events

        # Group by binned feature and calculate counts
        grouped = df.groupby('binned_feature')[target].agg(['count', 'sum']).reset_index()
        grouped.rename(columns={'count': 'total', 'sum': 'events'}, inplace=True)

        # Calculate non-events
        grouped['non_events'] = grouped['total'] - grouped['events']

        # Check for zero counts to avoid division by zero
        grouped.replace({'events': {0: np.nan}, 'non_events': {0: np.nan}}, inplace=True)

        # Drop any rows with NaN values (events or non_events)
        grouped.dropna(subset=['events', 'non_events'], inplace=True)

        # Calculate WoE and avoid division by zero
        grouped['woe'] = np.log((grouped['events'] / total_events) / (grouped['non_events'] / total_non_events))

        # Calculate bad probability for plotting
        grouped['bad_probability'] = grouped['events'] / grouped['total']

        # Create a plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar plot for counts
        ax1.bar(grouped['binned_feature'].astype(str), grouped['events'] / grouped['total'], color='lightcoral', alpha=0.7, label='bad')
        ax1.bar(grouped['binned_feature'].astype(str), grouped['non_events'] / grouped['total'], 
                bottom=grouped['events'] / grouped['total'], color='lightblue', alpha=0.7, label='good')

        # Twin axis for WoE line plot
        ax2 = ax1.twinx()
        ax2.plot(grouped['binned_feature'].astype(str), grouped['woe'], color='blue', marker='o', label='WoE')

        # Format the plot
        ax1.set_title(f'{feature} (IV={total_events:.3f})', fontsize=14)
        ax1.set_ylabel('Bin count distribution', fontsize=12)
        ax2.set_ylabel('Bad probability', fontsize=12)
        ax1.set_xticklabels(grouped['binned_feature'].astype(str), rotation=45, fontsize=10)

        # Adding the percentage labels on bars
        for index, row in grouped.iterrows():
            ax1.text(index, row['events'] / row['total'] + row['non_events'] / row['total'] / 2, 
                    f"{row['events'] / row['total']:.1%}, {row['total']}", ha='center', fontsize=10)

        # Legends
        ax1.legend(loc='upper left', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)

        # Style adjustments
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(0.8)
        ax1.spines['bottom'].set_linewidth(0.8)
        
        plt.tight_layout()
        plt.show()

    def save_merged_data(self, final_merged_df, output_file, file_path):
        # Ensure the output directory exists
        os.makedirs(file_path, exist_ok=True)  
        
        # Save the merged DataFrame to CSV
        final_merged_df.to_csv(f"{file_path}/{output_file}.csv", index=False)
        print(f"Merged data saved to {file_path}/{output_file}.csv")