

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #to get best K neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
import random
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
    

def rigid_regulation(data,test_data):
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import RidgeCV
    X_train,y_train,X_test= pre.create_sets(data,test_data)
    
    alpha_values = [0.1, 1, 10, 100]  # Example alpha values to try

    ridge_cv = RidgeCV(alphas=alpha_values, cv=5)  # Use 5-fold cross-validation
    ridge_cv.fit(X_train, y_train)  # X is your input data, y is your target variable

    best_alpha = ridge_cv.alpha_
    print(f"Best alpha value: {best_alpha}")

    # Once you have the best alpha value, you can fit the model with the entire dataset
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = ridge.predict(X_test)
    # # Save the prediction in a CSV file
    creation_result_file(y_pred,'prediction_L2.csv')
    
def lasso_regulation(data,test_data):
    from sklearn.linear_model import Lasso, LassoCV
    X_train,y_train,X_test= pre.create_sets(data,test_data)
    alpha_values = [0.1, 1, 10, 100]  # Example alpha values to try

    lasso_cv = LassoCV(alphas=alpha_values, cv=5)  # Use 5-fold cross-validation
    lasso_cv.fit(X_train, y_train)  # X is your input data, y is your target variable

    best_alpha = lasso_cv.alpha_
    print(f"Best alpha value: {best_alpha}")

    # Once you have the best alpha value, you can fit the model with the entire dataset
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X_train, y_train)
    y_pred= lasso.predict(X_test)
    # # Save the prediction in a CSV file
    creation_result_file(y_pred,'prediction_L1.csv')


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def gradient_descent(data, test_data, learning_rate=0.01, epochs=1000):
    """
    Perform linear regression using gradient descent.

    Parameters:
    - X: ECFP fingerprint features (numpy array or pandas DataFrame)
    - y: Retention time values (numpy array or pandas Series)
    - learning_rate: Learning rate for gradient descent
    - epochs: Number of training iterations

    Returns:
    - weights: Learned weights for the linear regression model
    """

    X_train,y_train,X_test= pre.create_sets(data,test_data)
    
    # Standardize features
    scaler_train = StandardScaler()
    X_scaled_train = scaler_train.fit_transform(X_train)
    
    # Add a column of ones for the bias term
    X_scaled_train = np.c_[np.ones(X_scaled_train.shape[0]), X_scaled_train]

    # Initialize weights
    weights = np.zeros(X_scaled_train.shape[1])

    # Gradient Descent
    for epoch in range(epochs):
        predictions = np.dot(X_scaled_train, weights)
        errors = predictions - y_train
        gradient = 2 * np.dot(errors, X_scaled_train) / len(y_train)
        weights -= learning_rate * gradient



    X_scaled_test = scaler_train.transform(X_test)
    
    # Add a column of ones for the bias term
    X_scaled_test = np.c_[np.ones(X_scaled_test.shape[0]), X_scaled_test]

    # Make predictions on the test set
    y_pred = np.dot(X_scaled_test, weights)

    # Save predictions to a file
    creation_result_file(y_pred, 'prediction_GD.csv')

    return y_pred
def artificial_neurons(data,test_data):
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_squared_error
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from skorch import NeuralNetRegressor #sklearn + pytorch
    X_train,y_train,X_test= pre.create_sets(data,test_data)

    # Standardize the input features
    standardizer = StandardScaler()
    X_standardized = standardizer.fit_transform(X_train)
    Y_standardized = standardizer.fit_transform(X_test)
    X_tensor = torch.tensor(X_standardized, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    #print(X_train.shape())
    print ("")
    print(X_tensor.size()) 
    # Define the neural network model using PyTorch
    class NN_model(nn.Module):
        def __init__(self, input_size=1040, n_neurons=8, dropout_rate=0.5):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, n_neurons),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_neurons, 1)
            )

        def forward(self, x):
            return self.layers(x)

    # create model with skorch
    model_skorch = NeuralNetRegressor(
        NN_model,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        max_epochs=400,#1000
        batch_size=64,#32
        verbose=False
    )

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'module__n_neurons': [8],
        'module__dropout_rate': [0, 0.1]
    }

    # Perform GridSearchCV for hyperparameter tuning
    print("gridsearchCV")
    grid_search = GridSearchCV(estimator=model_skorch, param_grid=param_grid, cv=3)
    print ("fitting grid")
    grid_result = grid_search.fit(X_tensor, y_tensor)
    print ("fitting done")

    print("Best MSE: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # Get the best model from the grid search
    mach2 = grid_result.best_estimator_

    # Fit the best model to the data
    print ("fitting model")
    mach2.fit(X_tensor, y_tensor)

    # Make predictions
    print("setting tensor")
    # Convert test data to tensor or numpy array
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)  # Assuming test_data is a DataFrame
    print("predicting")
    y_pred_torch = mach2.predict(X_test_tensor)
    #print (y_pred_torch.shape())
    print(type(y_pred_torch))
    
    # Save predictions to a file
    print("creating file")
    creation_result_file(y_pred_torch, 'artificial_neurons.csv')