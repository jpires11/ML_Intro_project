import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetRegressor #sklearn + pytorch
from skorch.callbacks import EarlyStopping


import prediction_functions as pf
import prepocessing as pre
def artificial_network(data,test_data):

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

    # Standardize input of test set
    standardized_test = X_standardizer.transform(X_test)
 
    # Use tensors   
    X_tensor = torch.tensor(X_standardized, dtype=torch.float32)
    y_tensor = torch.tensor(Y_standardized, dtype=torch.float32)
    test_tensor = torch.tensor(standardized_test, dtype=torch.float32)
   
    activation_functions = [
    ('ReLU', nn.ReLU()),
    ('LeakyReLU', nn.LeakyReLU()),
    ('Tanh', nn.Tanh()),
    ('Sigmoid', nn.Sigmoid()),
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

        def forward(self, x):
            return self.layers(x)
        
    
    # create model with skorch
    from skorch.callbacks import EarlyStopping
    model_skorch = NeuralNetRegressor(
        NN_model,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        optimizer__lr=0.001,
        max_epochs=400,
        batch_size=32,
        callbacks=[EarlyStopping(patience=15)],  # Adjust patience 
        verbose=True
        
    )

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'module__n_neurons': [50,100,256],
        'module__dropout_rate':[0.2,0.5],
        'module__activation': [func for name, func in activation_functions],
        #'optimizer': [optim.Adam, optim.SGD, optim.RMSprop],
        #'optimizer__lr': [0.001, 0.01,0.1],
        #'module__l1_strength': [0.001, 0.01],
        #'optimizer__weight_decay': [0.0001, 0.001, 0.01], # L2 strength
        #'max_epochs': [ 300,400] 
        #'callbacks': [[EarlyStopping(patience=10)] for i in range(5, 15)]  # Patience values to search
    }

    # Perform GridSearchCV for hyperparameter tuning
    print("gridsearchCV")
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
    print("predicting")
    y_pred_torch = mach2.predict(test_tensor)
    
    # Inverse transform the standardized predictions
   # y_pred_standardized = y_pred_torch.flatten()  # Flatten predictions
    y_pred_standardized = y_standardizer.inverse_transform(y_pred_torch).flatten()
    y_pred = y_pred_standardized.flatten()  # Assuming 'y_pred_standardized' contains the inverse transformed values

    # Save predictions to a file
    print("creating file")
    pf.creation_result_file(y_pred, 'neural_network.csv')