from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
def creation_result_file(prediction, name_of_file):
    ids = range(2, len(prediction) + 2)  # Generate IDs starting from 2
    output_df = pd.DataFrame({'ID': ids, 'RT': prediction})

    # Save the DataFrame to a CSV file in the folder results
    output_df.to_csv(os.path.join("Results",name_of_file), index=False)
    
    
def linear_model(data,test_data):
    
    
    X = data[[f'ECFP_{i}' for i in range(1, 1025)]]  # Adjust column names accordingly
    y = data['RT']
    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Set the training DataSet
    X_train = data[[f'ECFP_{i}' for i in range(1, 1025)]]  # Adjust column names accordingly
    y_train = data['RT']
    X_test = test_data[[f'ECFP_{i}' for i in range(1, 1025)]]  # Adjust column names accordingly
    
    # Initialize and train the linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = linear_model.predict(X_test)
    
    # Save the predication in csv file
    creation_result_file(y_pred,'prediction_linear_model.csv')
    #test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #print(f"Test RMSE: {test_rmse:.4f}")