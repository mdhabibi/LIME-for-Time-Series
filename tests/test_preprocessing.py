import pandas as pd
import numpy as np
from src.data_preprocess import load_data, check_and_clean_data, split_data, preprocess_data

def test_load_data():
    """This test should verify that data is loaded correctly and is of type DataFrame.""" 
    data = load_data("data/ecg_train.csv")
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"

def test_check_and_clean_data():
    """This test verifies that rows with null or infinite values are removed."""
    df = pd.DataFrame({
        'A': [1, 2, np.nan, np.inf],
        'B': [np.inf, 3, 4, 4]
    })
    cleaned_data = check_and_clean_data(df)
    assert cleaned_data.isnull().any().any() == False, "There should be no null values"
    assert np.isinf(cleaned_data).any().any() == False, "There should be no infinite values"
    assert len(cleaned_data) == 1, "Only one row should remain after cleaning"

def test_split_data():
    """This test checks if the dataset is correctly split into features and targets."""
    df = pd.DataFrame({
        'target': [0, 1, 1, 0],
        'feature1': [1, 2, 3, 4],
        'feature2': [5, 6, 7, 8]
    })
    features, target = split_data(df, target_column='target', split=True)
    assert 'target' not in features.columns, "The target column should not be in features"
    assert all(target == df['target']), "The target series should match the target column"

def test_preprocess_data_no_split():
    # Test preprocessing without splitting.
    data = preprocess_data("data/ecg_train.csv", split=False)
    assert isinstance(data, pd.DataFrame), "Preprocessed data should be a pandas DataFrame"
    assert data.isnull().any().any() == False, "There should be no null values after preprocessing"
    assert np.isinf(data).any().any() == False, "There should be no infinite values after preprocessing"
