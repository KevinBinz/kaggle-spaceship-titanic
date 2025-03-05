import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to Python path if not already there
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from feature_engineer import FeatureEngineer

@pytest.fixture
def toy_train_data():
    """Create a toy training dataset."""
    data = {
        'PassengerId': ['0001_01', '0002_01', '0003_01', '0004_01', '0005_01'],
        'HomePlanet': ['Earth', 'Mars', 'Europa', 'Earth', 'Europa'],
        'CryoSleep': [True, False, True, False, True],
        'Cabin': ['F/1/S', 'B/2/P', 'C/3/S', 'D/4/P', 'E/5/S'],
        'Destination': ['TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e', 'TRAPPIST-1e', 'PSO J318.5-22'],
        'Age': [39, 24, 58, 33, 16],
        'VIP': [False, False, True, False, True],
        'RoomService': [0, 109, 43, 0, 303],
        'FoodCourt': [0, 9, 3576, 1283, 70],
        'ShoppingMall': [0, 25, 0, 371, 151],
        'Spa': [0, 549, 6715, 3329, 565],
        'VRDeck': [0, 44, 49, 193, 2],
        'Name': ['Willy Santantines', 'Alfred Beston', 'Betty Johnson', 'Mary Smith', 'John Doe'],
        'Transported': [False, True, False, True, True]
    }
    df = pd.DataFrame(data)
    
    # Split Cabin and Name columns
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = df['CabinNum'].astype(int)
    df[['FirstName', 'LastName']] = df['Name'].str.split(' ', expand=True)
    df['CabinNumLen'] = df['CabinNum'].astype(str).str.len()
    
    # Add ZeroExpense column
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['ZeroExpense'] = (df[expense_cols] == 0).all(axis=1)
    
    return df

@pytest.fixture
def toy_test_data():
    """Create a toy test dataset."""
    data = {
        'PassengerId': ['0005_01', '0006_01', '0007_01', '0008_01', '0009_01'],
        'HomePlanet': ['Europa', 'Mars', 'Earth', 'Europa', 'Earth'],
        'CryoSleep': [True, False, True, False, True],
        'Cabin': ['B/1/P', 'C/2/S', 'D/3/P', 'E/4/S', 'F/5/P'],
        'Destination': ['PSO J318.5-22', '55 Cancri e', 'TRAPPIST-1e', 'PSO J318.5-22', '55 Cancri e'],
        'Age': [16, 44, 26, 32, 38],
        'VIP': [True, False, True, False, True],
        'RoomService': [303, 0, 181, 0, 199],
        'FoodCourt': [70, 9, 0, 1283, 0],
        'ShoppingMall': [151, 25, 0, 371, 0],
        'Spa': [565, 549, 0, 3329, 0],
        'VRDeck': [2, 44, 0, 193, 0],
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Erraiam Flatic', 'Alice Brown']
    }
    df = pd.DataFrame(data)
    
    # Split Cabin and Name columns
    df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)
    df['CabinNum'] = df['CabinNum'].astype(int)
    df[['FirstName', 'LastName']] = df['Name'].str.split(' ', expand=True)
    df['CabinNumLen'] = df['CabinNum'].astype(str).str.len()
    
    # Add ZeroExpense column
    expense_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    df['ZeroExpense'] = (df[expense_cols] == 0).all(axis=1)
    
    return df

def test_create_cabin_features(toy_train_data):
    """Test cabin feature creation."""
    fe = FeatureEngineer()
    df = fe.create_cabin_features(toy_train_data.copy())
    
    # Check if new columns are created
    assert 'CabinRegion' in df.columns
    assert 'CabinSize' in df.columns
    
    # Check if CabinRegion has correct number of categories
    assert len(df['CabinRegion'].cat.categories) == 7
    
    # Check if CabinSize is calculated correctly
    assert df['CabinSize'].min() == 1
    assert df['CabinSize'].max() == 1

def test_create_age_features(toy_train_data):
    """Test age feature creation."""
    fe = FeatureEngineer()
    df = fe.create_age_features(toy_train_data.copy())
    
    # Check if new columns are created
    assert 'AgeDecile' in df.columns
    
    # Check if AgeDecile has correct number of categories
    assert len(df['AgeDecile'].cat.categories) == 10

def test_create_family_features(toy_train_data):
    """Test family feature creation."""
    fe = FeatureEngineer()
    df = fe.create_family_features(toy_train_data.copy())
    
    # Check if new columns are created
    assert 'FamilySize' in df.columns
    
    # Check if FamilySize is calculated correctly
    assert df['FamilySize'].min() == 1
    assert df['FamilySize'].max() == 1

def test_create_transport_features(toy_train_data):
    """Test transport feature creation."""
    fe = FeatureEngineer()
    df = fe.create_transport_features(toy_train_data.copy())
    
    # Check if new columns are created
    assert 'Ysum' in df.columns
    assert 'Ysize' in df.columns
    assert 'Ypct' in df.columns
    
    # Check if transport features are calculated correctly
    assert df['Ysum'].min() >= 0
    assert df['Ysize'].min() >= 1
    assert df['Ypct'].min() >= 0
    assert df['Ypct'].max() <= 1

def test_impute_missing_values(toy_test_data):
    """Test missing value imputation."""
    fe = FeatureEngineer()
    df = toy_test_data.copy()

    # Convert boolean columns to nullable boolean type first
    df['VIP'] = df['VIP'].astype('boolean')
    df['CryoSleep'] = df['CryoSleep'].astype('boolean')

    # Add some missing values
    df.loc[0, 'HomePlanet'] = pd.NA
    df.loc[1, 'Destination'] = pd.NA
    df.loc[2, 'VIP'] = pd.NA
    df.loc[3, 'Age'] = pd.NA
    df.loc[4, 'CryoSleep'] = pd.NA

    df = fe.impute_missing_values(df)
    
    # Check if missing values are imputed
    assert not df['HomePlanet'].isna().any()
    assert not df['Destination'].isna().any()
    assert not df['VIP'].isna().any()
    assert not df['Age'].isna().any()
    assert not df['CryoSleep'].isna().any()

def test_engineer_features(toy_train_data, toy_test_data):
    """Test the complete feature engineering pipeline."""
    fe = FeatureEngineer()

    # Process both datasets
    train_engineered = fe.engineer_features(toy_train_data.copy())
    test_engineered = fe.engineer_features(toy_test_data.copy())
    
    # Check if all expected columns are present
    expected_columns = [
        'CabinRegion', 'CabinSize', 'AgeDecile', 'FamilySize',
        'Ysum', 'Ysize', 'Ypct'
    ]
    
    for col in expected_columns:
        if col in ['Ysum', 'Ysize', 'Ypct']:  # These only exist in training data
            assert col in train_engineered.columns
            assert col not in test_engineered.columns
        else:
            assert col in train_engineered.columns
            assert col in test_engineered.columns

if __name__ == '__main__':
    pytest.main([__file__, '-v']) 