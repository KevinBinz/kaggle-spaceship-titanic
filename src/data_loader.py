import pandas as pd
import numpy as np
import zipfile
import os
from logging_utils import log

class DataLoader:
    def __init__(self, train_path='train.csv', test_path='test.csv', zip_path='spaceship-titanic.zip'):
        """Initialize the DataLoader with paths to train and test data."""
        self.train_path = train_path
        self.test_path = test_path
        self.zip_path = zip_path
        self.train_raw = None
        self.test_raw = None

    def _ensure_data_files(self):
        if not os.path.exists(self.zip_path):
            log.info("\nDownloading competition data from Kaggle...")
            os.system('kaggle competitions download -c spaceship-titanic')
            log.info("Data download complete.")

        """Ensure data files exist by unzipping if necessary."""
        if not (os.path.exists(self.train_path) and os.path.exists(self.test_path)):
            log.info(f"Data files not found. Unzipping {self.zip_path}...")
            if not os.path.exists(self.zip_path):
                raise FileNotFoundError(f"Neither data files nor zip file found. Please ensure {self.zip_path} exists.")
            
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall('.')
            log.info("Data files extracted successfully.")

    def load_data(self):
        """Load the raw training and test data."""
        self._ensure_data_files()
        self.train_raw = pd.read_csv(self.train_path)
        self.test_raw = pd.read_csv(self.test_path)
        return self.train_raw, self.test_raw

    def get_nan_counts(self, df):
        """Count NaN values in each column of the dataframe."""
        return df.isna().sum()

    def split_cabin(self, df):
        """Split the Cabin column into Deck, CabinNum, and Side."""
        df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True, n=3)
        df['CabinNumLen'] = df['CabinNum'].str.len()
        df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
        return df

    def split_name(self, df):
        """Split the Name column into FirstName and LastName."""
        df[['FirstName', 'LastName']] = df['Name'].str.split(' ', expand=True, n=2)
        return df

    def extract_group_info(self, df):
        """Extract group number from PassengerId and calculate group-related features."""
        df['GroupNum'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
        df['GroupSize'] = df.groupby(['GroupNum'])['GroupNum'].transform('size')
        return df

    def calculate_expenditure(self, df):
        """Calculate total expenditure and related features."""
        expense_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df[expense_columns] = df[expense_columns].fillna(0)
        df['Expenditure'] = df[expense_columns].sum(axis=1)
        df['LogExpenditure'] = np.log(df['Expenditure'] + 1)
        df['ZeroExpense'] = df['Expenditure'] == 0
        return df

    def process_data(self):
        """Process both training and test data with all transformations."""
        if self.train_raw is None or self.test_raw is None:
            self.load_data()

        for df in [self.train_raw, self.test_raw]:
            df = self.split_cabin(df)
            df = self.split_name(df)
            df = self.extract_group_info(df)
            df = self.calculate_expenditure(df)

        self.train_processed = self.train_raw
        self.test_processed = self.test_raw
        return self.train_processed, self.test_processed 