

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #to get best K neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm # used in poisson regression
import numpy as np
import random
import pandas as pd
import os
import prepocessing as pre
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor #sklearn + pytorch
from skorch.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb




def creation_result_file(prediction, name_of_file):
    """
    Creates a result file from a prediction array.

    Args:
    - prediction (array-like): The predicted values.
    - name_of_file (str): The name of the file to be created.

    Returns:
    None

    This function generates a result file from a prediction array and saves it as a CSV file
    in the 'Results' folder. The prediction array is adjusted to ensure non-negative values,
    and the output file contains IDs starting from 1 and the corresponding predicted values.
    """
    prediction[prediction < 0] = 0
    ids = range(1, len(prediction) + 1)  # Generate IDs starting from 1
    output_df = pd.DataFrame({'ID': ids, 'RT': prediction})

    # Save the DataFrame to a CSV file in the folder results
    output_df.to_csv(os.path.join("Results",name_of_file), index=False)
    
    
def linear_model(X_train,y_train,X_test):
    """
    Trains a linear regression model using the provided training data and predicts on the test set.

    Args:
    - X_train (array-like): Training input samples.
    - y_train (array-like): Target values for training.
    - X_test (array-like): Test input samples for prediction.

    Returns:
    None

    This function initializes and trains a linear regression model using the given training data
    (X_train, y_train) and predicts on the provided test set (X_test). The predictions are saved
    in a CSV file named 'prediction_linear_model.csv' using the `creation_result_file` function.
    """
    np.random.seed(42)
    #Setup the training and test sets
    # Initialize and train the linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = linear_model.predict(X_test)
    
    # Save the predication in csv file
    creation_result_file(y_pred,'prediction_linear_model.csv')
    
def knn_regression_best_model(X_train,y_train):
    """
    Finds the optimal number of neighbors for KNN regression and returns the associated MSE.

    Args:
    - X_train (array-like): Training input samples.
    - y_train (array-like): Target values for training.

    Returns:
    int: The best number of neighbors found through cross-validation.

    This function splits the data into training and holdout validation sets, performs a grid search
    with cross-validation to find the optimal number of neighbors for KNN regression, and returns
    the best number of neighbors found. It prints the best number of neighbors and the associated
    mean squared error (MSE) on the holdout validation set.

    Note:
    This function internally uses scikit-learn's GridSearchCV to perform hyperparameter tuning
    for the KNN regressor.
    """
    np.random.seed(42)
    # Split data into training and holdout validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
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


def knn_regression(X_train,y_train,X_test):
    """
    Performs KNN regression on given training and test data.

    Args:
    - X_train (array-like): Training input samples.
    - y_train (array-like): Target values for training.
    - X_test (array-like): Test input samples for prediction.

    Returns:
    None

    This function initializes and performs KNN regression on the provided training data (X_train, y_train)
    and predicts on the given test set (X_test). The number of neighbors is determined using the
    `knn_regression_best_model` function, and predictions are saved in a CSV file named
    'prediction_knn.csv' using the `creation_result_file` function.
    """
    np.random.seed(42)
    # Initialize KNN Regressor
    n_neighbors = knn_regression_best_model(X_train,y_train)
    knn = KNeighborsRegressor(n_neighbors)  

    # Fit the KNN model and Predict 'y' for the test set
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # # Save the prediction in a CSV file
    creation_result_file(y_pred,'prediction_knn.csv')
    

def ridge_regulation(X_train,y_train,X_test):
    """
    Performs Ridge regression with cross-validated alpha selection.

    Args:
    - X_train (array-like): Training input samples.
    - y_train (array-like): Target values for training.
    - X_test (array-like): Test input samples for prediction.

    Returns:
    None

    This function conducts Ridge regression on the provided training data (X_train, y_train) using
    cross-validated alpha selection. It identifies the best alpha value using RidgeCV with a specified
    set of alpha values and 5-fold cross-validation. The trained model is then applied to predict
    on the given test set (X_test). The predictions are saved in a CSV file named 'prediction_L2.csv'
    using the `creation_result_file` function.
    """
    
    np.random.seed(42)
    alpha_values = [0.1, 1, 10, 100]  # Example alpha values to try
    ridge_cv = RidgeCV(alphas=alpha_values, cv=5)  # Use 5-fold cross-validation
    ridge_cv.fit(X_train, y_train)  # X is your input data, y is your target variable
    print("Best MSE: %f using %s" % (ridge_cv.best_score_, ridge_cv.alpha_))
    best_alpha = ridge_cv.alpha_

    # Once you have the best alpha value, you can fit the model with the entire dataset
    ridge = Ridge(alpha=best_alpha)
    ridge.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = ridge.predict(X_test)
    # # Save the prediction in a CSV file
    creation_result_file(y_pred,'prediction_L2.csv')
    
    
def lasso_regulation(X_train,y_train,X_test):
    """
    Trains a Lasso regression model with hyperparameter tuning and predicts on the test set.

    Args:
    - X_train : Training input features.
    - y_train : Target values for training.
    - X_test : Test input features for prediction.

    Returns:
    None

    This function performs Lasso regression on the provided training data (X_train, y_train).
    It conducts hyperparameter tuning using cross-validation and a predefined set of alpha values.
    The model is trained with the best alpha value obtained from the tuning process and used to predict
    on the test set. The predictions are saved in a CSV file named 'prediction_L1.csv' using the
    `creation_result_file` function.
    """
    np.random.seed(42)  # Set seed for reproducibility
    alpha_values = [0.1, 1, 5,10,15,20,25,50, 100] 
    
    # Hyperparameter tuning with cross validation
    lasso_cv = LassoCV(alphas=alpha_values, cv=5)  
    # Fitting of the model
    lasso_cv.fit(X_train, y_train) 
    best_alpha = lasso_cv.alpha_
    print(f"Best alpha value: {best_alpha}")

    # Fit the model with best parameters
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(X_train, y_train)
    y_pred= lasso.predict(X_test)
    # # Save the prediction in a CSV file
    creation_result_file(y_pred,'prediction_L1.csv')



def gradient_descent(X_train,y_train,X_test, learning_rate=0.01, epochs=1000, save = True):
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

    np.random.seed(42)  # Set seed for reproducibility
    # Standardize features
    scaler_train = StandardScaler()
    X_scaled_train = scaler_train.fit_transform(X_train)
    X_scaled_test = scaler_train.transform(X_test)
    # Add a column of ones for the bias term
    X_scaled_train = np.c_[np.ones(X_scaled_train.shape[0]), X_scaled_train]
    X_scaled_test = np.c_[np.ones(X_scaled_test.shape[0]), X_scaled_test]
    # Initialize weights
    weights = np.zeros(X_scaled_train.shape[1])

    # Gradient Descent
    for epoch in range(epochs):
        predictions = np.dot(X_scaled_train, weights)
        errors = predictions - y_train
        gradient = 2 * np.dot(errors, X_scaled_train) / len(y_train)
        weights -= learning_rate * gradient

    # Make predictions on the test set
    y_pred = np.dot(X_scaled_test, weights)

    # Save predictions to a file
    if(save == True):
        creation_result_file(y_pred, 'prediction_GD.csv')

    return y_pred



def artificial_neurons(X_train,y_train,X_test, use_grid_search=True):
    """
    Trains a neural network model using Skorch and performs predictions on the test set.

    Args:
    - X_train : Training input samples.
    - y_train : Target values for training.
    - X_test : Test input samples for prediction.
    - use_grid_search (bool, optional): Flag to perform grid search for hyperparameter tuning. Default is True.

    Returns:
    None

    This function builds a neural network model using PyTorch and Skorch. It standardizes the target values,
    defines various activation functions, and creates a neural network model architecture with configurable
    hyperparameters such as the number of neurons, dropout rate, activation functions, and L1 regularization.
    It utilizes Skorch's `NeuralNetRegressor` along with `GridSearchCV` for hyperparameter tuning if
    `use_grid_search` is True. The predictions on the test set are saved in a CSV file named 'artificial_neurons.csv'
    using the `creation_result_file` function.
    """
    # Set seed for NumPy
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    # Standardization 
    
    y_standardizer = StandardScaler()
    y_train = y_standardizer.fit_transform(y_train.values.reshape(-1,1))
    # Tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)

 
    activation_functions = [
    ('ReLU', nn.ReLU()),
    ('LeakyReLU', nn.LeakyReLU()),
    ('Tanh', nn.Tanh()),
    ('Sigmoid', nn.Sigmoid()),
    ]

    # Define the neural network model using PyTorch
    class NN_model(nn.Module):
        def __init__(self, input_size=X_train.shape[1], n_neurons=8, dropout_rate=0.5,activation=nn.ReLU(), l1_strength=0.001):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, n_neurons),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_neurons, n_neurons),  # Second hidden layer
                activation,
                nn.Dropout(p=dropout_rate),
                nn.Linear(n_neurons, 1)
            )
            self.l1_strength = l1_strength

        def forward(self, x):
            return self.layers(x)
        
        def l1_penalty(self):
            l1_reg = 0
            for param in self.parameters():
                l1_reg += torch.norm(param, p=1)
            return self.l1_strength * l1_reg
        
    # create model with skorch
    model_skorch = NeuralNetRegressor(
        NN_model,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        optimizer__lr=0.001,
        module__l1_strength=0.001,# Adjust the L1 strength value
        max_epochs=400,
        batch_size=32,
        callbacks=[EarlyStopping(patience=20)],  # Adjust patience 
        verbose=True
    )

    if not use_grid_search:
        # Use predefined hyperparameters
        best_params = {
            'module__n_neurons': 1024,
            'module__dropout_rate': 0.2,
            'module__activation': nn.Sigmoid(),
            'module__l1_strength': 0.001,
            
        }

        # Set the predefined hyperparameters
        model_skorch.set_params(**best_params)
        print("Using predefined hyperparameters:")
        print(best_params)
    else:
        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'module__n_neurons': [256, 512, 1024],
            'module__dropout_rate': [0.2, 0.5],
            'module__activation': [func for name, func in activation_functions],
            'module__l1_strength': [0.001, 0.01],
        }

        # Perform GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model_skorch, param_grid=param_grid, cv=3,n_jobs=-1,scoring="neg_mean_squared_error")
        grid_result = grid_search.fit(X_tensor, y_tensor)

        # Get the best model from the grid search and fit it
        model_skorch = grid_result.best_estimator_
        print("Best MSE: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        
    model_skorch.fit(X_tensor, y_tensor)
    
    
    # Make predictions and inverse the standardisation of the output
    y_pred = model_skorch.predict(test_tensor)
    y_pred = y_standardizer.inverse_transform(y_pred.reshape(-1, 1)).flatten()

    # Save predictions to a file
    creation_result_file(y_pred, 'artificial_neurons.csv')
    
def forest(X_train,y_train,X_test, use_grid_search=True):
    """
    Trains a Random Forest regressor and predicts on the test set.

    Args:
    - X_train : Training input features.
    - y_train : Target values for training.
    - X_test : Test input features for prediction.
    - use_grid_search (bool, optional): Flag to perform grid search for hyperparameter tuning. Default is True.

    Returns:
    None

    This function builds a Random Forest regressor model using the provided training data (X_train, y_train).
    It standardizes the input features, and if `use_grid_search` is True, it performs hyperparameter tuning using
    GridSearchCV. If not, it uses predefined hyperparameters. The predictions on the test set are saved in a
    CSV file named 'random_forest.csv' using the `creation_result_file` function.
    """
    # Standardize the input features
    X_standardizer = StandardScaler()
    X_train = X_standardizer.fit_transform(X_train)
    X_test = X_standardizer.transform(X_test)
    
    # Set seed for reproducibility
    np.random.seed(42)
    model = RandomForestRegressor(random_state=42)
    if not use_grid_search:
        # Use predefined hyperparameters
        best_params = {
            'n_estimators': 200,
            'max_depth': None,
            'max_features': None
        }

        # Set the predefined hyperparameters
        model.set_params(**best_params)
        print("Using predefined hyperparameters:")
        print(best_params)
    else:
        # Define the parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [ 200, 400, 600],  # Number of trees in the forest
            'max_depth': [None, 5, 10],  # Maximum depth of the tree
            'max_features': ['log2', None]  # Max features to consider for splitting
        }
        # Tuning of the hyperparameters
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error',verbose =1)
        grid_search.fit(X_train, y_train)
        print("Best MSE: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
        model = grid_search.best_estimator_
    # Prediction
    y_pred = model.predict(X_test)
    
    # Save predictions to a file
    creation_result_file(y_pred, 'random_forest.csv')


def xgb_predict(X_train, y_train, X_test, use_grid_search=True):
    """
    Trains an XGBoost regressor and predicts on the test set.

    Args:
    - X_train : Training input features.
    - y_train : Target values for training.
    - X_test : Test input features for prediction.
    - use_grid_search (bool, optional): Flag to perform grid search for hyperparameter tuning. Default is True.

    Returns:
    array-like: Predicted values.

    This function builds an XGBoost regressor model using the provided training data (X_train, y_train).
    It allows hyperparameter tuning using GridSearchCV if `use_grid_search` is True, otherwise, it uses predefined
    hyperparameters. The predictions on the test set are saved in a CSV file named 'XGB.csv' using the
    `creation_result_file` function.
    """
    # Initialize XGBoost regressor or classifier based on the problem
    model = xgb.XGBRegressor()  # For regression, change to XGBClassifier for classification

    if not use_grid_search:
        # Hyperparameters of your choice
        best_params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 2000,
            'reg_alpha': 0.01,
            'min_child_weight': 1,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
        }
        
        # Use predefined hyperparameters
        model.set_params(**best_params)
        print("Using predefined hyperparameters:")
        print(best_params)
    else:
        # Hyperparameter grid for tuning
        param_grid = {
            'max_depth': [ 5, 7],  
            'learning_rate': [ 0.1],
            'n_estimators': [1000,1500,2000],
            'reg_alpha': [ 0.01],
            'min_child_weight': [1],
            'subsample': [0.6],
            'colsample_bytree': [0.6],
        }

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        print("Best MSE: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

        # Get the best model
        best_model = grid_search.best_estimator_

        # Train the best model
        best_model.fit(X_train, y_train)

        # Assign the best model to the 'model' variable
        model = best_model

    # Train the model with either the predefined or best hyperparameters
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)
    print("creating file")
    creation_result_file(y_pred, 'XGB.csv')

    return y_pred
