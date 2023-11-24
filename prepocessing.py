
import pandas as pd
import os
def test_missing_values():
    """testing the data for missing values
    """
    # Load your test dataset
    data = pd.read_csv('train.csv')  # Replace with your test dataset

    # Check for missing values
    missing_values = data.isnull().sum()

    # Assert that no missing values should be present in the dataset
    assert missing_values.sum() == 0, f"Missing values found: {missing_values}"
    
    
def preprocess_and_check_constants(data):
    # Step 1: Remove constant columns
    non_constant_cols = data.columns[data.nunique() > 1]
    data_no_const = data[non_constant_cols]

    # Step 2: Check which columns were removed
    constant_cols = data.columns.difference(non_constant_cols)

    # Print columns that are deleted
    print("Deleted Constant Columns:", constant_cols.tolist())

    # Save the modified DataFrame to a new CSV file
    data_no_const.to_csv(os.path.join("Data_Set", 'preprocessed_data.csv'), index=False)

    return data_no_const


def remove_highly_correlated(data, threshold=0.9):
    import numpy as np
    
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    # Calculate the correlation matrix for numeric columns
    corr_matrix = data[numeric_cols].corr().abs()

    # Calculate the correlation matrix
    corr_matrix = data.corr().abs()

    # Create a mask to identify highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Print highly correlated columns before removal
    print("Highly Correlated Columns:", to_drop)

    # Drop the highly correlated columns
    data_no_corr = data.drop(columns=to_drop)
    data_no_corr.to_csv(os.path.join("Data_Set", 'preprocessed_data.csv'), index=False)
    
    return data_no_corr
