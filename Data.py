
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


data, test_data, train_preprocessed, test_preprocessed = pre.preprocess()


def excel_doc():
    
    # Export the subset data to an Excel file [Only the first 10 rows] // might need to install Excel viewer extention in vs code 
    data.head(10).to_excel(os.path.join("Excel", 'table_of_data.xlsx'), index=False)  
    test_data.head(10).to_excel(os.path.join("Excel", 'table_of_test_data.xlsx'), index=False)
    train_preprocessed.head(10).to_excel(os.path.join("Excel", 'table_processed_train.xlsx'), index=False)
    test_preprocessed.head(10).to_excel(os.path.join("Excel", 'table_processed_test.xlsx'), index=False)

"""n =25

dV.scatterRTvsCompound(data, n)
dV.RTvsCDDD(dV.mergeRT_CDDD(data, cddd, 100))
dV.HeatMap(pre.mergeRT_CDDD(data, cddd))
dV.RTvsCompoundbyLab(data, n, save = True)"""

#pre.preprocess_and_check_constants(train_preprocessed)
test_preprocessed=test_preprocessed.drop(columns=pre.remove_highly_correlated(train_preprocessed))
train_clean = pd.read_csv(os.path.join("Data_Set",'preprocessed_data.csv'))
#pf.knn_regression(train_clean,test_preprocessed)

pf.rigid_regulation(train_clean,test_preprocessed)

#pf.lasso_regulation(train_clean,test_preprocessed)

#pf.DG_regression_best_model(data, train_clean,test_preprocessed)

#dV.GD_parameters(train_clean, test_data, save = True)
#pf.gradient_descent(train_clean,test_preprocessed, learning_rate=0.05, epochs=400)

pf.artificial_neurons(train_clean,test_preprocessed)

#pf.forest(train_clean,test_preprocessed)
