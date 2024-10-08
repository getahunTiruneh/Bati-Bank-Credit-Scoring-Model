import numpy as np
import pandas as pd
import scorecardpy as sc
from monotonic_binning.monotonic_woe_binning import Binning  

class ModelEvaluator:
    def __init__(self, df, target):
        """
        Initialize the ModelEvaluator class with a DataFrame and the target column.
        Args:
            df (pd.DataFrame): The dataset containing features and target.
            target (str): The target column name (binary classification).
        """
        self.df = df
        self.target = target
        self.breaks = {}

    def woe_num(self):
        """
        Calculate optimal binning breaks for numerical features using monotonic binning.
        
        Returns:
            dict: A dictionary containing binning breaks for each numerical feature.
        """
        numerical_features = ['Total_Transaction_Amount', 'Avg_Transaction_Amount',
                            'Transaction_Count', 'Std_Transaction_Amount']

        for col in numerical_features:
            # Check for NaN or inf values before fitting
            if self.df[col].isnull().any() or np.isinf(self.df[col]).any():
                print(f"Warning: {col} contains NaN or infinite values. Please clean your data.")
                continue
            
            bin_object = Binning(self.target, n_threshold=50, y_threshold=10, p_threshold=0.35, sign=False)
            bin_object.fit(self.df[[self.target, col]])

            # Ensure breaks are properly formatted
            self.breaks[col] = sorted(set(bin_object.bins[1:-1]))  # Sort and deduplicate
            
        return self.breaks

    def adjust_woe(self):
        """
        Adjust the Weight of Evidence (WoE) calculation for numerical features 
        based on the calculated breaks and plot the results.
        """
        if not self.breaks:
            raise ValueError("No breaks have been calculated. Please run woe_num() first.")
        # Adjust WoE calculation using breaks
        bins_adj = sc.woebin(self.df, y=self.target, breaks_list=self.breaks, positive='1')  # Use '1' for "Bad"
        
        # Plot the WoE values
        sc.woebin_plot(bins_adj)

    def woeval(self, train):
        """
        Apply WoE transformation to a given DataFrame.
        
        Args:
            train (pd.DataFrame): The input DataFrame for WoE transformation.  
        Returns:
            pd.DataFrame: The transformed DataFrame with WoE values.
        """
        if not self.breaks:
            raise ValueError("No breaks have been calculated. Please run woe_num() first.")
        
        if not isinstance(train, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        # WoE transformation
        bins_adj = sc.woebin(self.df, y=self.target, breaks_list=self.breaks, positive='1')
        train_woe = sc.woebin_ply(train, bins_adj)
        return train_woe

    def calculate_iv(self, df_merged, y):
        """
        Calculate the Information Value (IV) for features in the provided DataFrame.
        Args:
            df_merged (pd.DataFrame): The merged DataFrame for which IV needs to be calculated.
            y (str): The target variable name.
            
        Returns:
            tuple: Cleaned DataFrame and a DataFrame containing the IV for each feature.
        """
        df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]  # Remove duplicated columns
        
        # Remove unnecessary columns
        df_merged1 = df_merged.drop(['CustomerId'], axis=1)

        # Calculate Information Value (IV)
        iv_results = sc.iv(df_merged1, y=y) 
        return df_merged, iv_results
