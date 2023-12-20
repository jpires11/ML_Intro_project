
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import os

def create_sets(ECFP = True, CDDD = False):
    '''
    Function:
    creates X_train, y_train, X_test sets by preprocessing the dataframes
    imported in the preprocess function. Standardises?
    Arguments: 
    ECFP and RT -- Booleans indicating what molecule representation we want in our sets.
    Return:
    X_train,y_train,X_test -- The arguments of the prediction functions
    '''
    train, test, data, test_data = preprocess(ECFP, CDDD)
    X_train = data.drop(["SMILES", 'RT', "mol", "Compound"], axis=1).copy()
    y_train = data['RT']
    X_test = test_data.drop(["SMILES", "mol", "Compound"], axis=1).copy()
    
<<<<<<< HEAD
    # Standardize the input features
    X_standardizer = StandardScaler()
    X_train = X_standardizer.fit_transform(X_train)
    # Standardize the output features (y_train)
    #y_standardizer = StandardScaler()
    #y_train_reshaped = y_train.values.reshape(-1, 1)  # Convert to NumPy array and reshape
    #y_train = y_standardizer.fit_transform(y_train_reshaped)
    X_test = X_standardizer.transform(X_test)
    
    
=======


>>>>>>> bbb7ffefd30d54b0fb8dbee8176debd7e54c35d8
    return X_train,y_train,X_test

def test_missing_values(data):
    """testing the data for missing values
    """
    # Load your test dataset
     

    # Check for missing values
    missing_values = data.isnull().sum()

    # Assert that no missing values should be present in the dataset
    assert missing_values.sum() == 0, f"Missing values found: {missing_values}"


def dummies(data, name):
    '''
    Function:
    Encodes Labs in binary.
    Arguments: 
    data -- The Panda Dataframes for which we wish to encode the Labs column
    name -- string indicating the name of the output csv file
    Return:
    data_encode -- New Panda Dataframes with encoded Lab columns.
    '''
    import os
    data_encode= data.copy()
    # Convert 'Lab' column to categorical if it's not already categorical
    data_encode['Lab'] = data_encode['Lab'].astype('category')
    
    # Convert 'Compound' column to categorical if it's not already categorical
    #data['Compound'] = data['Compound'].astype('category')

    # Get dummies for 'Lab' and 'Compound' columns
    encoded_cols_lab = pd.get_dummies(data_encode['Lab'], prefix='Lab')
    ##encoded_cols_compound = pd.get_dummies(data['Compound'], prefix='Compound')
    
    # Convert False and True values to binary (0 and 1)
    encoded_cols_lab.replace({True: 1, False: 0}, inplace=True)
    #encoded_cols_compound.replace({True: 1, False: 0}, inplace=True)

    # Drop the original 'Lab' and 'Compound' columns from the DataFrame
    #data = data.drop(['Lab', 'Compound'], axis=1)
    data_encode = data_encode.drop(['Lab'], axis=1)
    # Concatenate the original DataFrame with the encoded columns
    data_encode = pd.concat([data_encode, encoded_cols_lab], axis=1)
    #data = pd.concat([data, encoded_cols_lab, encoded_cols_compound], axis=1)
    
    # Save the modified DataFrame to a CSV file
    data_encode.to_csv(os.path.join("Data_set", name), index=False)
    
    return data_encode

def preprocess_and_check_constants(data, test_data):
    '''
    Function:
    Removes constant colums in a dataset.
    Arguments: 
    data, test_data -- The Panda Dataframes for which we wish to remove the constant columns
    Return:
    data, test_data -- New Panda Dataframes without constant columns.
    '''
    # Identify constant columns in data
    constant_cols_data = data.columns[data.nunique() == 1].tolist()

    # Print columns that are constant in data
    print("Constant Columns in Training Data:", constant_cols_data)

    # Remove constant columns from data
    data = data.drop(constant_cols_data, axis=1)

    # Remove the same columns from test data
    test_data = test_data.drop(constant_cols_data, axis=1, errors='ignore')
    return data,test_data

#test if variable are correlated 
def remove_highly_correlated(data,test_data, threshold=0.9):
    '''
    Function:
    Removes Highly correlated colums in a data set.
    Arguments: 
    data, test_data -- The Panda Dataframes for which we wish to remove the correlated columns
    threshold -- float indicating the correlation threshold
    Return:
    data, test_data -- New Panda Dataframes without correlated columns.
    '''
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
    data = data.drop(columns=to_drop)
    data.to_csv(os.path.join("Data_set", 'train_modified_data_CDDD.csv'), index=False)
    test_data = test_data.drop(columns=to_drop)
    test_data.to_csv(os.path.join("Data_set", 'test_modified_data_CDDD.csv'), index=False)
    return data,test_data

def mergeRT_CDDD(data, cddd, RT = True, ECFP = False):
    '''
    Function:
    Merges CDDDs to dataset. Keeps ECFPs in the dataset as well is ECFP is True
    Arguments: 
    ECFP and RT -- Booleans indicating what molecule representation we want in our sets.
    cddd --  Dataframe containing CDDD values
    data -- A Panda Dataframe containing ECFPs, Labs and Compounds
    Return:
    subdet_data. A dataframe containing the desired representations.
    '''
<<<<<<< HEAD
    merged_data = pd.merge(data, cddd, on='SMILES',how = "left")
=======
    merged_data = pd.merge(data, cddd, on='SMILES', how='left')
>>>>>>> bbb7ffefd30d54b0fb8dbee8176debd7e54c35d8
    # Select columns of interest
    if RT == True:
        selected_columns = ['RT'] + ['SMILES'] + ['mol'] + ['Compound'] + ["Lab"] + [f'cddd_{i}' for i in range(1, 513)]
    else:
        # to create validation sets
        selected_columns = ['SMILES'] + ['mol'] + ['Compound'] + ["Lab"] + [f'cddd_{i}' for i in range(1, 513)]
    if ECFP == True:
        selected_columns += [f'ECFP_{i}' for i in range(1, 1025)]
    subset_data = merged_data[selected_columns]
    return subset_data

def preprocess(ECFP = True, CDDD = False):
    '''
    Function:
    Imports datasets train and test. Replaces missing CDDD values with the average of the column.
    Calls dummies function to encode Labs using 0s and 1s. Removes constants and highly correlated values.
    Arguments: 
    ECFP and RT -- Booleans indicating what molecule representation we want in our sets.
    Return:
    train_data, test_data, train_preprocessed, test_preprocessed -- The preprcocessed dataframes.
    '''
    train_data= pd.read_csv(os.path.join("Data_set",'train.csv'))
    test_data=pd.read_csv(os.path.join("Data_set","test.csv"))
    
    if CDDD == True:
        cddd = pd.read_csv(os.path.join("Data_set",'cddd.csv'))
    
        # Merge of cddd into the datasets based on smiles
        train_data = mergeRT_CDDD(train_data, cddd, ECFP = ECFP)
        test_data = mergeRT_CDDD(test_data, cddd, RT = False, ECFP = ECFP)
        # Fill the missing values in the dataset with the mean of the columns 
        columns_starting_with_cddd_test = [col for col in test_data.columns if col.startswith('cddd_')]
        for col in columns_starting_with_cddd_test:
            col_mean = test_data[col].mean()
            test_data[col].fillna(col_mean, inplace=True)
            
        columns_starting_with_cddd_train = [col for col in train_data.columns if col.startswith('cddd_')]
        for col in columns_starting_with_cddd_train:
            col_mean = train_data[col].mean()
            train_data[col].fillna(col_mean, inplace=True)
    
    # Encoding of Labs
    train_preprocessed=dummies(train_data,'train_modified_data_CDDD.csv')
    test_preprocessed=dummies(test_data,'test_modified_data_CDDD.csv')
    # Removal of constant or correlated parameters
    train_preprocessed, test_preprocessed = preprocess_and_check_constants(train_preprocessed,test_preprocessed)
    train_preprocessed, test_preprocessed = remove_highly_correlated(train_preprocessed,test_preprocessed, threshold=0.9)

    return train_data, test_data, train_preprocessed, test_preprocessed
