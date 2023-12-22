

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

file_list = [os.path.join("Results","prediction_GD.csv"),
             os.path.join("Results","XGB.csv"),  
             os.path.join("Results","prediction_knn.csv"),
             os.path.join("Results","prediction_L1.csv"),
             os.path.join("Results","prediction_linear_model.csv"),
             os.path.join("Results","random_forest.csv"),
             os.path.join("Results","artificial_neurons.csv")]



def visualise():
    data = pd.read_csv("Data_set/train.csv") 
    cddd = pd.read_csv("Data_set/cddd.csv")
    #dV.scatterRTvsCompound(data, 25)
    #dV.RTvsCDDD(pre.mergeRT_CDDD(data, cddd))
    #dV.HeatMap(pre.mergeRT_CDDD(data, cddd))
    dV.RTvsCompoundbyLab(data, 25, save = True)
    #dV.compare_predictions(file_list, save = True)
    
#visualise()


def repruduce_results():
    X_train,y_train,X_test = pre.create_sets(ECFP = False, CDDD = True)
    pf.artificial_neurons(X_train,y_train,X_test,False)
    pf.xgb_predict(X_train,y_train,X_test,False)
    
    #pf.forest(X_train,y_train,X_test,False)
    #pf.gradient_descent(X_train,y_train,X_test, learning_rate=0.01, epochs=400)
    #pf.linear_model(X_train,y_train,X_test)
    #pf.knn_regression(X_train,y_train,X_test)
    #pf.ridge_regulation(X_train,y_train,X_test)
    #pf.lasso_regulation(X_train,y_train,X_test)
    
    "some visalisation"
    #dV.GD_parameters(X_train,y_train,X_test, save = True)
    #dV.excel_doc(data,test_data, train_preprocessed, test_preprocessed,name_train_processed="train_processed.xlsx", name_test_processed="test_processed.xlsx")
    #dV.excel_doc(data,test_data, train_preprocessed, test_preprocessed,"train_processed_just_CDDD.xlsx","test_processed_just_CDDD.xlsx")

repruduce_results()