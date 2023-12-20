
import dataVis as dV
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os #to save plots
from openpyxl.workbook import Workbook
import prediction_functions as pf
import prepocessing as pre 
import numpy as np

X_train,y_train,X_test = pre.create_sets(ECFP = False, CDDD = True)
#data, test_data, train_preprocessed, test_preprocessed = pre.preprocess(False, True)
#dataC, test_dataC, train_preprocessedC, test_preprocessedC = pre.preprocess(True, False)
#dV.excel_doc(data,test_data, train_preprocessed, test_preprocessed,name_train_processed="train_processed.xlsx", name_test_processed="test_processed.xlsx")
#dV.excel_doc(data,test_data, train_preprocessed, test_preprocessed,"train_processed_just_CDDD.xlsx","test_processed_just_CDDD.xlsx")
"""
dV.scatterRTvsCompound(data, 25)
dV.RTvsCDDD(dV.mergeRT_CDDD(data, cddd, 100))
dV.HeatMap(pre.mergeRT_CDDD(data, cddd))
dV.RTvsCompoundbyLab(data, 25, save = True)
"""
#pf.knn_regression(train_clean,test_preprocessed)

#pf.ridge_regulation(X_train,y_train,X_test)

#pf.lasso_regulation(train_preprocessed,test_preprocessed)

#pf.DG_regression_best_model(data, train_clean,test_preprocessed)

#dV.GD_parameters(train_clean, test_data, save = True)
#pf.gradient_descent(train_clean,test_preprocessed, learning_rate=0.05, epochs=400)

#pf.ridge_regulation(X_train,y_train,X_test)
pf.artificial_neurons(X_train,y_train,X_test)
#pf.forest(X_train,y_train,X_test)
#pf.xgb_predict(X_train,y_train,X_test)


#pf.forest(X_train,y_train,X_test)
#pf.new_forest(train_preprocessed,test_preprocessed)
#pf.NN_prediction(train_preprocessed, test_preprocessed)

#nn.artificial_network(train_preprocessed,test_preprocessed)

pf.xgb_predict(X_train,y_train,X_test)
'''
XGBC = pf.xgb_predict(train_preprocessedC,test_preprocessedC)
plt.scatter(np.arange(1375), XGBF)
plt.scatter(np.arange(1375), XGBC)
plt.show()'''
#dV.compare_predictions(os.path.join("Results","XGB.csv"), os.path.join("Results","artificial_neurons.csv"))