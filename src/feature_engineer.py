import pandas as pd
import numpy as np
import inspect
import ast
import textwrap

class FeatureEngineer:
    def __init__(self):
        """Initialize the FeatureEngineer."""
        pass

    def _analyze_function_calls(self, method):
        """Extract function calls from a method's source code.
        
        Args:
            method: The method to analyze
            
        Returns:
            list: List of function names called within the method
        """
        # Get the method's source code lines
        source_lines, _ = inspect.getsourcelines(method)
        
        # Join the lines and dedent
        source = textwrap.dedent(''.join(source_lines))
        
        # Parse the source code into an AST
        tree = ast.parse(source)
        
        # Find all function calls
        function_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Handle method calls (e.g., self._extract_cabin_components)
                    if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                        function_calls.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    # Handle direct function calls
                    function_calls.append(node.func.id)
        
        return sorted(set(function_calls))  # Remove duplicates and sort

    def _get_function_calls(self, method_name):
        """Extract function calls from a method by name.
        
        Args:
            method_name (str): Name of the method to analyze
            
        Returns:
            list: List of function names called within the method
        """
        method = getattr(self.__class__, method_name)
        return self._analyze_function_calls(method)

    def get_used_functions(self):
        """Get lists of functions used in preprocessing and feature engineering.
        
        Returns:
            tuple: (preprocessing_functions, engineering_functions)
                - preprocessing_functions: List of functions called in preprocess_raw_data
                - engineering_functions: List of functions called in engineer_features
        """
        preprocessing_functions = self._get_function_calls('preprocess_raw_data')
        engineering_functions = self._get_function_calls('engineer_features')
        return preprocessing_functions, engineering_functions

    def preprocess_raw_data(self, df):
        """Apply basic preprocessing to raw data."""
        df = self._extract_cabin_components(df)
        df = self._extract_name_components(df)
        df = self._extract_group_components(df)
        df = self._calculate_spending_features(df)
        return df

    def _extract_cabin_components(self, df):
        """Extract and process cabin-related components."""
        df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True, n=3)
        df['CabinNumLen'] = df['CabinNum'].str.len()
        df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')
        return df

    def _extract_name_components(self, df):
        """Extract and process name-related components."""
        df[['FirstName', 'LastName']] = df['Name'].str.split(' ', expand=True, n=2)
        return df

    def _extract_group_components(self, df):
        """Extract and process group-related components."""
        df['GroupNum'] = df['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int)
        df['GroupSize'] = df.groupby(['GroupNum'])['GroupNum'].transform('size')
        return df

    def _calculate_spending_features(self, df):
        """Calculate spending-related features."""
        expense_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        df[expense_columns] = df[expense_columns].fillna(0)
        df['Expenditure'] = df[expense_columns].sum(axis=1)
        df['LogExpenditure'] = np.log(df['Expenditure'] + 1)
        df['ZeroExpense'] = df['Expenditure'] == 0
        return df

    def create_cabin_features(self, df):
        """Create derived features based on cabin information."""
        df['CabinRegion'] = pd.qcut(df['CabinNum'], q=7)
        df['CabinSize'] = df.groupby(['CabinNum'])['CabinNum'].transform('size')
        return df

    def create_age_features(self, df):
        """Create derived features based on age."""
        df['AgeDecile'] = pd.qcut(df['Age'], q=10)
        return df

    def create_family_features(self, df):
        """Create derived features based on family information."""
        df['FamilySize'] = df.groupby(['LastName'])['LastName'].transform('size')
        return df

    def create_transport_features(self, df):
        """Create derived features related to transport status (for training data only)."""
        if 'Transported' in df.columns:
            df['Ysum'] = df.groupby(['LastName'])['Transported'].transform('sum')
            df['Ysize'] = df.groupby(['LastName'])['Transported'].transform('size')
            df['Ypct'] = df['Ysum'].div(df['Ysize'])
        return df

    def impute_missing_values(self, df):
        """Impute missing values using various strategies."""
        # HomePlanet imputation based on Deck
        df.loc[(df['HomePlanet'].isna()) & (df['Deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet'] = 'Europa'
        df.loc[(df['HomePlanet'].isna()) & (df['Deck'] == 'G'), 'HomePlanet'] = 'Earth'
        
        # Destination imputation
        df.loc[df['Destination'].isna(), 'Destination'] = 'TRAPPIST-1e'
        
        # VIP imputation
        df.loc[df['VIP'].isna(), 'VIP'] = False
        
        # Age imputation by subgroup median
        age_mask = df['Age'].isna()
        # First try imputing by HomePlanet and Deck
        df.loc[age_mask, 'Age'] = df.groupby(['HomePlanet', 'Deck'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )[age_mask]
        
        # If there are still missing ages, try imputing by HomePlanet only
        age_mask = df['Age'].isna()
        df.loc[age_mask, 'Age'] = df.groupby(['HomePlanet'])['Age'].transform(
            lambda x: x.fillna(x.median())
        )[age_mask]
        
        # If there are still missing ages, fill with overall median
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # CryoSleep imputation based on spending
        cryo_mask = df['CryoSleep'].isna()
        df.loc[cryo_mask, 'CryoSleep'] = df.groupby(['ZeroExpense'])['CryoSleep'].transform(
            lambda x: x.fillna(x.mode().iloc[0]).infer_objects(copy=False)
        )[cryo_mask]
        
        # Convert boolean columns back to boolean type
        df['VIP'] = df['VIP'].astype('boolean')
        df['CryoSleep'] = df['CryoSleep'].astype('boolean')
        
        return df

    def impute_missing_values_simple(self, df):
        """Impute missing values using simple strategies (mode for categorical, median for numeric)."""
        # Destination imputation using mode
        df.loc[df['Destination'].isna(), 'Destination'] = df['Destination'].mode().iloc[0]
        
        # VIP imputation using mode
        df.loc[df['VIP'].isna(), 'VIP'] = df['VIP'].mode().iloc[0]
        
        # CryoSleep imputation using mode
        df.loc[df['CryoSleep'].isna(), 'CryoSleep'] = df['CryoSleep'].mode().iloc[0]
        
        # Age imputation using median
        df['Age'] = df['Age'].fillna(df['Age'].median())
        
        # Convert boolean columns back to boolean type
        df['VIP'] = df['VIP'].astype('boolean')
        df['CryoSleep'] = df['CryoSleep'].astype('boolean')
        
        return df

    def engineer_features(self, df):
        """Apply all feature engineering transformations."""
        # First preprocess the raw data
        df = self.preprocess_raw_data(df)
        
        # Convert boolean columns to nullable boolean type
        df['VIP'] = df['VIP'].astype('boolean')
        df['CryoSleep'] = df['CryoSleep'].astype('boolean')
        
        # Create derived features
        df = self.create_cabin_features(df)
        df = self.create_age_features(df)
        df = self.create_family_features(df)
        df = self.create_transport_features(df)
        
        # Impute missing values using original strategy
        df = self.impute_missing_values(df)
        
        return df 