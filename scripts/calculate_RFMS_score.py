import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RFMSRiskClassifier:
    def __init__(self, df):
        self.df = df

    def calculate_recency(self, current_date):
        self.df['TransactionStartTime'] = pd.to_datetime(self.df['TransactionStartTime'], errors='coerce').dt.tz_localize(None)
        current_date = pd.to_datetime(current_date).tz_localize(None)
        self.df['Recency'] = (current_date - self.df['TransactionStartTime']).dt.days

    def calculate_frequency(self):
        self.df['Frequency'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')

    def calculate_monetary(self):
        self.df['Monetary'] = self.df.groupby('CustomerId')['Total_Transaction_Amount'].transform('sum')

    def calculate_seasonality(self):
        self.df['Seasonality'] = self.df.groupby('CustomerId')['Transaction_Month'].transform(lambda x: x.nunique())

    def normalize_rfms(self):
        rfms_columns = ['Recency', 'Frequency', 'Monetary', 'Seasonality']
        # Ensure RFMS columns are numeric
        self.df[rfms_columns] = self.df[rfms_columns].apply(pd.to_numeric, errors='coerce')
        self.df[rfms_columns] = self.df[rfms_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    def assign_risk_category(self, threshold=0.3):
        self.df['RFMS_score'] = self.df[['Recency', 'Frequency', 'Monetary', 'Seasonality']].mean(axis=1)
        self.df['Risk_category'] = self.df['RFMS_score'].apply(lambda x: 'good' if x >= threshold else 'bad')

    def woe_binning(self, n_bins=5, epsilon=1e-4):
        self.df['RFMS_bin'] = pd.qcut(self.df['RFMS_score'], q=n_bins, duplicates='drop')
        woe_df = self.df.groupby('RFMS_bin')['Risk_category'].value_counts(normalize=False).unstack().fillna(0)
        
        # Include RFMS_bin in the resulting DataFrame
        woe_df['RFMS_bin'] = woe_df.index.astype(str)  # Convert index to column
        woe_df['total_good'] = woe_df['good'].sum()
        woe_df['total_bad'] = woe_df['bad'].sum()
        
        woe_df['good_dist'] = (woe_df['good'] + epsilon) / (woe_df['total_good'] + epsilon)
        woe_df['bad_dist'] = (woe_df['bad'] + epsilon) / (woe_df['total_bad'] + epsilon)
        woe_df['WoE'] = np.log(woe_df['good_dist'] / woe_df['bad_dist'])
        woe_df['IV'] = (woe_df['good_dist'] - woe_df['bad_dist']) * woe_df['WoE']
        
        iv_total = woe_df['IV'].sum()
        
        # Return the DataFrame with 'RFMS_bin'
        return woe_df.reset_index(drop=True), iv_total

    def plot_woe(self, woe_df):
        # Ensure that the necessary columns are present
        required_columns = ['good', 'bad', 'WoE', 'RFMS_bin']
        for col in required_columns:
            if col not in woe_df.columns:
                raise ValueError(f"woe_df must contain '{col}' column for plotting.")

        # Convert the 'good', 'bad', and 'WoE' columns to numeric types if not already
        woe_df['good'] = pd.to_numeric(woe_df['good'], errors='coerce')
        woe_df['bad'] = pd.to_numeric(woe_df['bad'], errors='coerce')
        woe_df['WoE'] = pd.to_numeric(woe_df['WoE'], errors='coerce')

        # Reset index to ensure the bin labels are appropriate for plotting
        woe_df = woe_df.reset_index()

        plt.figure(figsize=(12, 6))

        # Stacked bar chart
        x = np.arange(len(woe_df))

        # Plotting the stacked bars for 'Good' and 'Bad'
        plt.bar(x, woe_df['good'], label='Good', color='#00CFFF', alpha=0.8)
        plt.bar(x, woe_df['bad'], bottom=woe_df['good'], label='Bad', color='#FF8C8C', alpha=0.8)

        # Plotting WoE line
        plt.twinx()
        sns.lineplot(x=x, y='WoE', data=woe_df, color='blue', marker='o', label='Weight of Evidence (WoE)', sort=False)

        # Title and labels
        plt.title('WoE Binning Plot for RFMS Score', fontsize=16)
        plt.xlabel('RFMS Score Bins', fontsize=12)
        plt.ylabel('Count Distribution', fontsize=12)
        plt.ylabel('Weight of Evidence (WoE)', fontsize=12)

        # Show legend
        plt.legend(loc='upper left')  # Position the legend outside the plot
        plt.grid()
        plt.xticks(ticks=x, labels=woe_df['RFMS_bin'].astype(str), rotation=45)
        plt.tight_layout()
        plt.show()

    def run(self, current_date):
        self.calculate_recency(current_date)
        self.calculate_frequency()
        self.calculate_monetary()
        self.calculate_seasonality()
        self.normalize_rfms()
        self.assign_risk_category()
        woe_results, iv_value = self.woe_binning()
        return self.df[['CustomerId', 'Recency', 'Frequency', 'Monetary', 'Seasonality', 'RFMS_score', 'Risk_category']], woe_results, iv_value
