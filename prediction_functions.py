

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
    prediction[prediction < 0] = 0
    ids = range(1, len(prediction) + 1)  # Generate IDs starting from 1
    output_df = pd.DataFrame({'ID': ids, 'RT': prediction})

    # Save the DataFrame to a CSV file in the folder results
    output_df.to_csv(os.path.join("Results",name_of_file), index=False)
    
    
def linear_model(data,test_data):
    
    np.random.seed(42)
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
    np.random.seed(42)
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
    np.random.seed(42)
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
    np.random.seed(42)
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
    np.random.seed(42)
    X_train,y_train,X_test= pre.create_sets(data,test_data)
    print(X_train)
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
    print (y_pred)
    # # Save the prediction in a CSV file
    creation_result_file(y_pred,'prediction_L2.csv')
    
def lasso_regulation(data,test_data):
    # Set seed for reproducibility
    from sklearn.linear_model import Lasso, LassoCV
    np.random.seed(42)
    X_train,y_train,X_test= pre.create_sets(data,test_data)
    alpha_values = [0.1, 1, 5,10,15,20,25,50, 100]  # Example alpha values to try

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
# Set seed for reproducibility
    np.random.seed(42)# Set seed for reproducibility

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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor #sklearn + pytorch
from skorch.callbacks import EarlyStopping
def artificial_neurons(data,test_data):

    # Set seed for NumPy
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    X_train,y_train,X_test= pre.create_sets(data,test_data)

    # Standardize the input features
    X_standardizer = StandardScaler()
    X_standardized = X_standardizer.fit_transform(X_train)

    # Standardize the output features (y_train)
    y_standardizer = StandardScaler()
    y_train_reshaped = y_train.values.reshape(-1, 1)  # Convert to NumPy array and reshape
    Y_standardized = y_standardizer.fit_transform(y_train_reshaped)


    standardized_test = X_standardizer.transform(X_test)
 
    
    X_tensor = torch.tensor(X_standardized, dtype=torch.float32)
    y_tensor = torch.tensor(Y_standardized, dtype=torch.float32)
    test_tensor = torch.tensor(standardized_test, dtype=torch.float32)


    print ("")
    print(X_tensor.size()) 
   
   
    activation_functions = [
    ('ReLU', nn.ReLU()),
    ('LeakyReLU', nn.LeakyReLU()),
    ('Tanh', nn.Tanh()),
    ('Sigmoid', nn.Sigmoid()),
    # Add more activation functions here
    ]

    # Define the neural network model using PyTorch
    class NN_model(nn.Module):
        def __init__(self, input_size=X_standardized.shape[1], n_neurons=8, dropout_rate=0.5,activation=nn.ReLU()):#, l1_strength=0.001):#, l2_strength=0.0001):
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
            #self.l1_strength = l1_strength
            #self.l2_strength = l2_strength

        def forward(self, x):
            return self.layers(x)
        
        """  def l1_penalty(self):
            l1_reg = 0
            for param in self.parameters():
                l1_reg += torch.norm(param, p=1)
            return self.l1_strength * l1_reg"""
        
        """ def l2_penalty(self):
            l2_reg = 0
            for param in self.parameters():
                l2_reg += torch.norm(param, p=2)
            return self.l2_strength * l2_reg"""

    # create model with skorch
    from skorch.callbacks import EarlyStopping
    model_skorch = NeuralNetRegressor(
        NN_model,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        optimizer__lr=0.001,
        #module__l1_strength=0.001,# Adjust the L1 strength value
        #optimizer__weight_decay=0.0001, #Adjust L2 strenght value
        max_epochs=30,
        batch_size=32,
        callbacks=[EarlyStopping(patience=15)],  # Adjust patience 
        verbose=True
        
    )

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'module__n_neurons': [8,50],
        'module__dropout_rate':[0.2,0.5]
        #'module__activation': [func for name, func in activation_functions],
        #'optimizer': [optim.Adam, optim.SGD, optim.RMSprop],
        #'optimizer__lr': [0.001, 0.01,0.1],
        #'module__l1_strength': [0.001, 0.01],
        #'optimizer__weight_decay': [0.0001, 0.001, 0.01], # L2 strength
        #'max_epochs': [ 300,400] 
        #'callbacks': [[EarlyStopping(patience=10)] for i in range(5, 15)]  # Patience values to search
    }

    # Perform GridSearchCV for hyperparameter tuning
    print("gridsearchCV")
    np.random.seed(42)
    grid_search = GridSearchCV(estimator=model_skorch, param_grid=param_grid, cv=10,n_jobs=-1,scoring="neg_mean_squared_error")
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
    print("predicting")
    y_pred_torch = mach2.predict(test_tensor)
    # Make predictions
    y_pred_standardized = y_pred_torch.flatten()  # Flatten predictions

    # Inverse transform the standardized predictions
    y_pred = y_standardizer.inverse_transform(y_pred_standardized.reshape(-1, 1)).flatten()

    # Save predictions to a file
    creation_result_file(y_pred, 'artificial_neurons.csv')

    
    y_pred_standardized = y_standardizer.inverse_transform(y_pred_torch).flatten()
    y_pred = y_pred_standardized.flatten()  # Assuming 'y_pred_standardized' contains the inverse transformed values

    # Save predictions to a file
    print("creating file")
    creation_result_file(y_pred, 'artificial_neurons.csv')
    
def forest(data,test_data):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.decomposition import PCA
    # Set seed for reproducibility
    np.random.seed(42)
    model = RandomForestRegressor(random_state=42)
    X_train,y_train,X_test= pre.create_sets(data,test_data)

    param_grid = {
        'n_estimators': [100,200,400],  # Number of trees in the forest
        'max_depth': [None,5,10,15],  # Maximum depth of the tree
        #'min_samples_split': [2, 5, 10],  # Test different values for min_samples_split
        #'min_samples_leaf': [1, 2, 4],  # Test different values for min_samples_leaf
        'max_features': ['sqrt', 'log2', None]  # Max features to consider for splitting
        #'bootstrap': [True, False],  # Whether bootstrap samples are used
       # 'max_samples': [0.5, 0.7, 0.9, None],  # Number of samples to draw for each tree
       # 'ccp_alpha': [0.0, 0.1, 0.2],  # Cost Complexity Pruning parameter
        #'max_leaf_nodes': [None, 10, 50, 100]  # Maximum number of leaf nodes in a tree
    }
    
 

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error',verbose =1)
    print ("fitting the grid")
    grid_search.fit(X_train, y_train)
    print("Best MSE: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
    
    best_model = grid_search.best_estimator_
    print("predicting")
    y_pred = best_model.predict(X_test)
    
    # Save predictions to a file
    print("creating file")
    creation_result_file(y_pred, 'random_forest.csv')

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

def xgb_predict(train_data, test_data):
    
    # Initialize XGBoost regressor or classifier based on the problem
    model = xgb.XGBRegressor()  # For regression, change to XGBClassifier for classification

    X_train, y_train, X_test = pre.create_sets(train_data, test_data)

    # Hyperparameter grid for tuning
    param_grid = {
       # 'max_depth': [3, 5, 7],
        #'learning_rate': [0.1, 0.01],
        #'n_estimators': [100,100,10000],
       # 'reg_alpha': [0, 0.001, 0.01],
        #'min_child_weight': [1, 3, 5],
        #'subsample': [0.6, 0.8, 1.0],
        #'colsample_bytree': [0.6, 0.8, 1.0],
        
        'max_depth': [ 7],
        'learning_rate': [0.1],
        #'n_estimators': [500],
        #'reg_alpha': [ 0.01],
        #
        #'min_child_weight': [1],
        #'subsample': [0.6],
        #'colsample_bytree': [0.6]
        
        # Add more parameters for tuning
    }

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1,verbose=1,scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    print("Best MSE: %f using %s" % (grid_search.best_score_, grid_search.best_params_))

    # Get the best model
    best_model = grid_search.best_estimator_

    # Train the best model
    best_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = best_model.predict(X_test)
    print("creating file")
    creation_result_file(y_pred, 'XGBß.csv')

    return y_pred
