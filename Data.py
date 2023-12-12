
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
import neural_network as nn


data, test_data, train_preprocessed, test_preprocessed = pre.preprocess(True, False)

#dV.excel_doc(data,test_data, train_preprocessed, test_preprocessed,"train_processed_just_CDDD.xlsx","test_processed_just_CDDD.xlsx")
"""n =25

dV.scatterRTvsCompound(data, n)
dV.RTvsCDDD(dV.mergeRT_CDDD(data, cddd, 100))
dV.HeatMap(pre.mergeRT_CDDD(data, cddd))
dV.RTvsCompoundbyLab(data, n, save = True)"""

#pre.preprocess_and_check_constants(train_preprocessed)
#test_preprocessed=test_preprocessed.drop(columns=pre.remove_highly_correlated(train_preprocessed))
#train_clean = pd.read_csv(os.path.join("Data_Set",'preprocessed_data.csv'))
#pf.knn_regression(train_clean,test_preprocessed)

#pf.rigid_regulation(train_preprocessed, test_preprocessed)
"""
train_preprocessed.fillna(0, inplace=True)
test_preprocessed.fillna(0, inplace=True)
"""

    
#pf.lasso_regulation(train_preprocessed,test_preprocessed)

#pf.DG_regression_best_model(data, train_clean,test_preprocessed)

#dV.GD_parameters(train_clean, test_data, save = True)
#pf.gradient_descent(train_clean,test_preprocessed, learning_rate=0.05, epochs=400)

#pf.artificial_neurons(train_preprocessed,test_preprocessed)

pf.forest(train_preprocessed,test_preprocessed)
#pf.new_forest(train_preprocessed,test_preprocessed)
#pf.NN_prediction(train_preprocessed, test_preprocessed)

#nn.artificial_network(train_preprocessed,test_preprocessed)