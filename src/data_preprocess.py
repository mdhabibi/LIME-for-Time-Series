import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_data(file_path, sep=',', header=None):
    """
    Load the dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
        sep (str): Separator used in the CSV file. Default is ','.
        header (int or None): Row number to use as the column names. Defaults to None.
    
    Returns:
        DataFrame: A pandas DataFrame containing the loaded data.
    """
    data = pd.read_csv(file_path, sep=sep, header=header).astype(float)
    return data

def check_and_clean_data(data):
    """
    Check for null and infinite values in the dataset and remove rows with such values.
    
    Parameters:
        data (DataFrame): The dataset to check and clean.
    
    Returns:
        DataFrame: The cleaned dataset.
    """
    # Check for and handle infinite values by removing rows
    if np.isinf(data.values).any():
        print("The dataset contains infinite values.")
        data = data.replace([np.inf, -np.inf], np.nan)  # Optionally replace infinities with NaN
        # Now remove rows with NaN values, which include those previously marked as infinities
        data = data.dropna()
    
    initial_row_count = data.shape[0]
    # Further clean if there are any remaining null values
    data = data.dropna()
    cleaned_row_count = data.shape[0]
    
    print(f"Removed {initial_row_count - cleaned_row_count} rows with null or infinite values.")
    
    return data


def split_data(data, target_column=0, split=False):
    """
    Optionally split the dataset into features and target.
    
    Parameters:
        data (DataFrame): The dataset to split.
        target_column (int): The index of the target column.
        split (bool): Whether to split the dataset into features and targets.
    
    Returns:
        DataFrame or (DataFrame, Series): If split is True, returns a tuple containing 
                                          the features DataFrame and the target Series.
                                          If split is False, returns the unmodified dataset.
    """
    if split:
        target = data.loc[:, target_column]
        features = data.drop(columns=[target_column])
        return features, target
    else:
        return data

def preprocess_data(file_path, sep=',', header=None, target_column=0, split=False):
    """
    Complete preprocessing workflow for loading, checking, cleaning,
    and optionally splitting a time series dataset from a CSV file.
    
    Parameters:
        file_path (str): Path to the CSV file.
        sep (str): Separator used in the CSV file. Default is ','.
        header (int or None): Row number to use as the column names, defaults to None.
        target_column (int): The index of the target column (only used if split is True).
        split (bool): Whether to split the dataset into features and targets.
    
    Returns:
        DataFrame or (DataFrame, Series): Depending on the split parameter, returns
                                           either the cleaned dataset or a tuple with
                                           features and targets.
    """
    data = load_data(file_path, sep=sep, header=header)
    data = check_and_clean_data(data)
    if split:
        return split_data(data, target_column=target_column, split=True)
    else:
        return data

def prepare_for_conv1d_training(features, labels, num_classes):
    """
    Prepares the dataset for Conv1D training: reshapes features and one-hot encodes labels.
    
    Parameters:
        features (DataFrame or ndarray): The features of the dataset.
        labels (Series or ndarray): The labels of the dataset.
        num_classes (int): The total number of classes in the dataset.
    
    Returns:
        tuple: A tuple containing the reshaped features and one-hot encoded labels.
    """
    # Reshape input data to be 3D [samples, timesteps, features] for Conv1D
    X_reshaped = np.expand_dims(features, axis=2)
    
    # Adjust labels to be zero-based if not already and one-hot encode
    labels_adjusted = labels - 1  # Adjust labels if they're not zero-based
    y_one_hot = to_categorical(labels_adjusted, num_classes=num_classes)
    
    return X_reshaped, y_one_hot
