

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os #to save plots
from openpyxl.workbook import Workbook
import dataVis as dV
import prediction_functions as pf
import prepocessing as pre 
import numpy as np

X_train,y_train,X_test = pre.create_sets(ECFP = True, CDDD = False)

#dV.excel_doc(data,test_data, train_preprocessed, test_preprocessed,name_train_processed="train_processed.xlsx", name_test_processed="test_processed.xlsx")
#dV.excel_doc(data,test_data, train_preprocessed, test_preprocessed,"train_processed_just_CDDD.xlsx","test_processed_just_CDDD.xlsx")
"""
dV.scatterRTvsCompound(data, 25)
dV.RTvsCDDD(dV.mergeRT_CDDD(data, cddd, 100))
dV.HeatMap(pre.mergeRT_CDDD(data, cddd))
dV.RTvsCompoundbyLab(data, 25, save = True)
"""
pf.linear_model(X_train,y_train,X_test)

#pf.poisson_regression(X_train,y_train,X_test)

#pf.knn_regression(X_train,y_train,X_test)

#pf.ridge_regulation(X_train,y_train,X_test)

#pf.lasso_regulation(X_train,y_train,X_test)

#pf.DG_regression_best_model(data, train_clean,test_preprocessed)

#dV.GD_parameters(train_clean, test_data, save = True)
pf.gradient_descent(X_train,y_train,X_test, learning_rate=0.01, epochs=400)

#pf.ridge_regulation(X_train,y_train,X_test)
#pf.artificial_neurons(X_train,y_train,X_test)
#pf.forest(X_train,y_train,X_test)
#pf.xgb_predict(X_train,y_train,X_test)


#pf.forest(X_train,y_train,X_test)
#pf.new_forest(train_preprocessed,test_preprocessed)
#pf.NN_prediction(train_preprocessed, test_preprocessed)

#nn.artificial_network(train_preprocessed,test_preprocessed)

##pf.xgb_predict(X_train,y_train,X_test,True)

file_list = [os.path.join("Results","prediction_GD.csv"), 
             os.path.join("Results","prediction_knn.csv"),
             os.path.join("Results","prediction_knn1.csv"),
             os.path.join("Results","prediction_L1.csv"),
             os.path.join("Results","prediction_linear_model.csv"),
             os.path.join("Results","prediction_poisson_model.csv"),
             os.path.join("Results","random_forest.csv"),
             os.path.join("Results","artificial_neurons.csv")]
dV.compare_predictions(file_list, save = True)