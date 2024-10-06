import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

class FeatureEngineering:
    def __init__(self, df):
        """
        Initializes the FeatureEngineering class with a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame to be processed.
        """
        self.df = df

    def remove_high_missing_values(self, threshold=0.5):
        """
        Removes columns with missing values exceeding the specified threshold.

        Args:
            threshold (float): The percentage threshold for missing values (default is 0.5).

        Returns:
            pd.DataFrame: DataFrame after dropping columns with high missing values.
        """
        missing_percentage = self.df.isnull().mean()
        if (missing_percentage > threshold).any():
            print("Columns with missing values more than threshold:")
            print(missing_percentage[missing_percentage > threshold])
            self.df.drop(columns=missing_percentage[missing_percentage > threshold].index, inplace=True)
        else:
            print("No columns found with missing values more than threshold.")
        return self.df

    def create_aggregate_features(self):
        """
        Creates aggregate features based on transaction data.

        Returns:
            pd.DataFrame: DataFrame with new aggregate feature columns.
        """
        self.df['Total_Transaction_Amount'] = self.df.groupby('CustomerId')['Amount'].transform('sum')
        self.df['Avg_Transaction_Amount'] = self.df.groupby('CustomerId')['Amount'].transform('mean')
        self.df['Transaction_Count'] = self.df.groupby('CustomerId')['TransactionId'].transform('count')
        self.df['Std_Transaction_Amount'] = self.df.groupby('CustomerId')['Amount'].transform('std')
        return self.df

    def extract_temporal_features(self):
        """
        Extracts temporal features from the transaction timestamp.

        Returns:
            pd.DataFrame: DataFrame with new temporal feature columns.
        """
        self.df['Transaction_Hour'] = self.df['TransactionStartTime'].dt.hour
        self.df['Transaction_Day'] = self.df['TransactionStartTime'].dt.day
        self.df['Transaction_Month'] = self.df['TransactionStartTime'].dt.month
        self.df['Transaction_Year'] = self.df['TransactionStartTime'].dt.year
        return self.df

    def handle_missing_values(self, method='imputation', strategy='mean', numerical_cols=None, categorical_cols=None):
        """
        Handles missing values in the DataFrame using specified methods.

        Args:
            method (str): The method to handle missing values ('imputation' or 'removal').
            strategy (str): The strategy for imputation ('mean', 'median', or 'mode').
            numerical_cols (list, optional): List of numerical columns to process.
            categorical_cols (list, optional): List of categorical columns to process.

        Returns:
            pd.DataFrame: DataFrame after handling missing values.
        """
        if numerical_cols is None:
            numerical_cols = self.df.select_dtypes(include=['float64', 'int']).columns.tolist()
        
        if categorical_cols is None:
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        # Handle numerical missing values
        if method == 'imputation':
            for col in numerical_cols:
                if strategy == 'mean':
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)
        elif method == 'removal':
            self.df.dropna(subset=numerical_cols, inplace=True)

        # Handle categorical missing values
        for col in categorical_cols:
            if method == 'imputation':
                self.df[col].fillna(self.df[col].mode().iloc[0], inplace=True)  # Filling with mode for categorical
            elif method == 'removal':
                self.df.dropna(subset=categorical_cols, inplace=True)

        return self.df

    def encode_categorical_variables(self, encoding_type='onehot'):
        """
        Encodes categorical variables using the specified encoding type.

        Args:
            encoding_type (str): The encoding method ('onehot' or 'label').

        Returns:
            pd.DataFrame: DataFrame after encoding categorical variables.
        """
        if encoding_type == 'onehot':
            self.df = pd.get_dummies(self.df, columns=['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId'], drop_first=False)
            # Ensure only boolean columns are converted to integers
            bool_cols = self.df.select_dtypes(include='bool').columns.tolist()
            self.df[bool_cols] = self.df[bool_cols].astype(int)
            # Rename the encoded columns for better readablity
            self.df.rename(columns={'ProviderId_ProviderId_1':'ProviderId1',
                                    'ProviderId_ProviderId_2':'ProviderId2',
                                    'ProviderId_ProviderId_3':'ProviderId3',
                                    'ProviderId_ProviderId_4':'ProviderId4',
                                    'ProviderId_ProviderId_5':'ProviderId5',
                                    'ProviderId_ProviderId_6':'ProviderId6',
                                    'ProductId_ProductId_1' :'ProductId1',
                                    'ProductId_ProductId_2' :'ProductId2',
                                    'ProductId_ProductId_3' :'ProductId3',
                                    'ProductId_ProductId_4' :'ProductId4',
                                    'ProductId_ProductId_5' :'ProductId5',
                                    'ProductId_ProductId_6' :'ProductId6',
                                    'ProductId_ProductId_7' :'ProductId7',
                                    'ProductId_ProductId_8' :'ProductId8',
                                    'ProductId_ProductId_9' :'ProductId9',
                                    'ProductId_ProductId_10' :'ProductId10',
                                    'ProductId_ProductId_11' :'ProductId11',
                                    'ProductId_ProductId_12' :'ProductId12',
                                    'ProductId_ProductId_13' :'ProductId13',
                                    'ProductId_ProductId_14' :'ProductId14',
                                    'ProductId_ProductId_15' :'ProductId15',
                                    'ProductId_ProductId_16' :'ProductId16',
                                    'ProductId_ProductId_19' :'ProductId19',
                                    'ProductId_ProductId_20' :'ProductId20',
                                    'ProductId_ProductId_21' :'ProductId21',
                                    'ProductId_ProductId_22' :'ProductId22',
                                    'ProductId_ProductId_23' :'ProductId23',
                                    'ProductId_ProductId_24' :'ProductId24',
                                    'ProductId_ProductId_27' :'ProductId27',
                                    'ChannelId_ChannelId_1':'ChannelId1',
                                    'ChannelId_ChannelId_2':'ChannelId2',
                                    'ChannelId_ChannelId_3':'ChannelId3',
                                    'ChannelId_ChannelId_5':'ChannelId5',}, inplace=True)
        elif encoding_type == 'label':
            le = LabelEncoder()
            categorical_cols = ['ProviderId', 'ProductId', 'ProductCategory', 'ChannelId']
            for col in categorical_cols:
                self.df[col] = le.fit_transform(self.df[col])
        
        return self.df

    def scale_numerical_features(self, scaling_type='normalization'):
        """
        Scales numerical features using the specified scaling method.

        Args:
            scaling_type (str): The scaling method ('normalization' or 'standardization').

        Returns:
            pd.DataFrame: DataFrame after scaling numerical features.
        """
        numeric_cols = ['Amount', 'Value', 'Total_Transaction_Amount', 'Avg_Transaction_Amount', 'Std_Transaction_Amount']
        
        if scaling_type == 'normalization':
            scaler = MinMaxScaler()
        elif scaling_type == 'standardization':
            scaler = StandardScaler()

        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self.df

    def run_feature_engineering(self):
        """
        Runs the full feature engineering pipeline on the DataFrame.

        Returns:
            pd.DataFrame: Final DataFrame after all feature engineering steps.
        """
        self.remove_high_missing_values(threshold=0.5)
        self.create_aggregate_features()
        self.extract_temporal_features()
        self.handle_missing_values(method='imputation', strategy='mean')
        self.encode_categorical_variables(encoding_type='onehot')
        self.scale_numerical_features(scaling_type='normalization')
        return self.df
    def save_processed_data(self, file_path):
        """
        Saves the processed DataFrame to a specified file path.

        Args:
            file_path (str): The file path to save the processed DataFrame.
        """
        self.df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")