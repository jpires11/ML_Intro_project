
import pandas as pd
import numpy as np
import os

def create_sets(data,test_data):
    X_train = data.drop(["SMILES",'RT',"mol","Compound"], axis=1)  # Adjust columns to drop if needed
    y_train = data['RT']
    X_test=test_data.drop(["SMILES","mol","Compound"], axis=1)
    return X_train,y_train,X_test

def test_missing_values():
    """testing the data for missing values
    """
    # Load your test dataset
    data = pd.read_csv('train.csv')  # Replace with your test dataset

    # Check for missing values
    missing_values = data.isnull().sum()

    # Assert that no missing values should be present in the dataset
    assert missing_values.sum() == 0, f"Missing values found: {missing_values}"


def dummies(data, name):
    import os
    # Convert 'Lab' column to categorical if it's not already categorical
    data['Lab'] = data['Lab'].astype('category')
    
    # Convert 'Compound' column to categorical if it's not already categorical
    #data['Compound'] = data['Compound'].astype('category')

    # Get dummies for 'Lab' and 'Compound' columns
    encoded_cols_lab = pd.get_dummies(data['Lab'], prefix='Lab')
    ##encoded_cols_compound = pd.get_dummies(data['Compound'], prefix='Compound')
    
    # Convert False and True values to binary (0 and 1)
    encoded_cols_lab.replace({True: 1, False: 0}, inplace=True)
    #encoded_cols_compound.replace({True: 1, False: 0}, inplace=True)

    # Drop the original 'Lab' and 'Compound' columns from the DataFrame
    #data = data.drop(['Lab', 'Compound'], axis=1)
    data = data.drop(['Lab'], axis=1)
    # Concatenate the original DataFrame with the encoded columns
    data = pd.concat([data, encoded_cols_lab], axis=1)
    #data = pd.concat([data, encoded_cols_lab, encoded_cols_compound], axis=1)
    
    # Save the modified DataFrame to a CSV file
    data.to_csv(os.path.join("Data_Set", name), index=False)

    
#test if there is constante values
    
def preprocess_and_check_constants(data):

    # Step 1: Remove constant columns
    non_constant_cols = data.columns[data.nunique() > 1]
    data_no_const = data[non_constant_cols]

    # Step 2: Check which columns were removed
    constant_cols = data.columns.difference(non_constant_cols)

    # Print columns that are deleted
    print("Deleted Constant Columns:", constant_cols.tolist())

    # Save the modified DataFrame to a new CSV file
    data_no_const.to_csv(os.path.join("Data_set", 'preprocessed_data.csv'), index=False)

    return data_no_const

#test if variable are correlated 
def remove_highly_correlated(data, threshold=0.9):
    # Identify only numeric columns
    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()

    # Calculate the correlation matrix for numeric columns
    corr_matrix = data[numeric_cols].corr().abs()

    # Create a mask to identify highly correlated features
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    # Print highly correlated columns before removal
    print("Highly Correlated Columns:", to_drop)

    # Drop the highly correlated columns
    data_no_corr = data.drop(columns=to_drop)
    data_no_corr.to_csv(os.path.join("Data_set", 'preprocessed_data.csv'), index=False)
    return  to_drop




"""standartisation"""
