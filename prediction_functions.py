

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #to get best K neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import os
import prepocessing as pre

def creation_result_file(prediction, name_of_file):
    ids = range(1, len(prediction) + 1)  # Generate IDs starting from 1
    output_df = pd.DataFrame({'ID': ids, 'RT': prediction})

    # Save the DataFrame to a CSV file in the folder results
    output_df.to_csv(os.path.join("Results",name_of_file), index=False)
    
    
def linear_model(data,test_data):
    
    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    #Setup the training and test sets
    X_train,y_train,X_test= pre.create_sets(data,test_data)
    # Initialize and train the linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = linear_model.predict(X_test)
    
    # Save the predication in csv file
    creation_result_file(y_pred,'prediction_linear_model.csv')
    #test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #print(f"Test RMSE: {test_rmse:.4f}")
    
    
import statsmodels.api as sm

def poisson_regression(data, test_data):
    
    #Setup the training and test sets
    X_train,y_train,X_test= pre.create_sets(data,test_data)
    # Fit the Poisson regression model
    poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()
    # Predict 'y' for the test set using the trained model
    y_pred = poisson_model.predict(X_test)
    # # Save the prediction in a CSV file
    creation_result_file(y_pred,'prediction_poisson_model.csv')
    
    
#KNN model
def knn_regression_best_model(data):
    """
    finding the best number of neighbors to use in KNN regression and also
    returns the mse useful to chose a cv appropriate
        
    """
    # Split data into training and holdout validation sets
    X = data.drop(["SMILES",'RT',"mol","Compound"], axis=1)  # Adjust columns to drop if needed
    y = data['RT']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define the range of hyperparameters to test
    param_grid = {'n_neighbors': [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]}  
    
    # Initialize KNN Regressor
    knn = KNeighborsRegressor()
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters and best model 
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)

    # Evaluate the best model on the holdout validation set using MSE
    mse = mean_squared_error(y_val, y_pred)
    print ("K is ",best_params['n_neighbors'])
    print()
    print ("MSE associated with the best number of neighbors ",mse)
    return best_params ['n_neighbors']

def knn_regression(data, test_data):

    #Setup the training and test sets
    X_train,y_train,X_test= pre.create_sets(data,test_data)
    
    # Initialize KNN Regressor
    n_neighbors = knn_regression_best_model(data)
    knn = KNeighborsRegressor(n_neighbors)  

    # Fit the KNN model and Predict 'y' for the test set
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # # Save the prediction in a CSV file
    creation_result_file(y_pred,'prediction_knn.csv')
    

    
    