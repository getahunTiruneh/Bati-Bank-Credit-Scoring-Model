import numpy as np
import pandas as pd
import scorecardpy as sc
from monotonic_binning.monotonic_woe_binning import Binning  
import os
import logging

# Create a 'log' folder if it doesn't exist
if not os.path.exists('../log'):
    os.makedirs('../log')

# Set up logging configuration
logging.basicConfig(
    filename='../log/model_evaluator.log',  # Log file path
    level=logging.DEBUG,  # Set the logging level to DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log message format
)

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
        logging.info('ModelEvaluator initialized with dataframe of shape %s and target column %s', df.shape, target)

    def woe_num(self):
        """
        Calculate optimal binning breaks for numerical features using monotonic binning.
        
        Returns:
            dict: A dictionary containing binning breaks for each numerical feature.
        """
        logging.info('Calculating WoE breaks for numerical features...')
        numerical_features = ['Total_Transaction_Amount', 'Avg_Transaction_Amount',
                            'Transaction_Count', 'Std_Transaction_Amount']

        for col in numerical_features:
            # Check for NaN or inf values before fitting
            if self.df[col].isnull().any() or np.isinf(self.df[col]).any():
                logging.warning("Warning: %s contains NaN or infinite values. Please clean your data.", col)
                continue

            try:
                bin_object = Binning(self.target, n_threshold=50, y_threshold=10, p_threshold=0.35, sign=False)
                bin_object.fit(self.df[[self.target, col]])

                # Ensure breaks are properly formatted
                self.breaks[col] = sorted(set(bin_object.bins[1:-1]))  # Sort and deduplicate
                logging.info('Binning successful for %s, breaks: %s', col, self.breaks[col])
            except Exception as e:
                logging.error('Error in binning for %s: %s', col, e)
        
        return self.breaks

    def adjust_woe(self):
        """
        Adjust the Weight of Evidence (WoE) calculation for numerical features 
        based on the calculated breaks and plot the results.
        """
        logging.info('Adjusting WoE calculations...')
        if not self.breaks:
            logging.error('No breaks have been calculated. Please run woe_num() first.')
            raise ValueError("No breaks have been calculated. Please run woe_num() first.")
        
        try:
            bins_adj = sc.woebin(self.df, y=self.target, breaks_list=self.breaks, positive='1')  # Use '1' for "Bad"
            sc.woebin_plot(bins_adj)
            logging.info('WoE adjustment successful and plot generated.')
        except Exception as e:
            logging.error('Error in adjusting WoE: %s', e)

    def woeval(self, train):
        """
        Apply WoE transformation to a given DataFrame.
        
        Args:
            train (pd.DataFrame): The input DataFrame for WoE transformation.  
        Returns:
            pd.DataFrame: The transformed DataFrame with WoE values.
        """
        logging.info('Applying WoE transformation...')
        if not self.breaks:
            logging.error('No breaks have been calculated. Please run woe_num() first.')
            raise ValueError("No breaks have been calculated. Please run woe_num() first.")
        
        if not isinstance(train, pd.DataFrame):
            logging.error("Input must be a pandas DataFrame. Provided type: %s", type(train))
            raise ValueError("Input must be a pandas DataFrame.")

        try:
            bins_adj = sc.woebin(self.df, y=self.target, breaks_list=self.breaks, positive='1')
            train_woe = sc.woebin_ply(train, bins_adj)
            logging.info('WoE transformation successful for the provided DataFrame.')
            return train_woe
        except Exception as e:
            logging.error('Error in applying WoE transformation: %s', e)

    def calculate_iv(self, df_merged, y):
        """
        Calculate the Information Value (IV) for features in the provided DataFrame.
        Args:
            df_merged (pd.DataFrame): The merged DataFrame for which IV needs to be calculated.
            y (str): The target variable name.
            
        Returns:
            tuple: Cleaned DataFrame and a DataFrame containing the IV for each feature.
        """
        logging.info('Calculating Information Value (IV) for features...')
        try:
            df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]  # Remove duplicated columns
            df_merged1 = df_merged.drop(['CustomerId'], axis=1)

            iv_results = sc.iv(df_merged1, y=y)
            logging.info('IV calculation successful for merged data.')
            return df_merged, iv_results
        except Exception as e:
            logging.error('Error in calculating IV: %s', e)
