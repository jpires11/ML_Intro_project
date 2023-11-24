
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

#load original data
data= pd.read_csv(os.path.join("Data_Set",'train.csv'))
test_data=pd.read_csv(os.path.join("Data_Set","test.csv"))
#process data and load it
pre.dummies(data,'train_modified_data.csv')
pre.dummies(test_data,'test_modified_data.csv')
train_preprocessed= pd.read_csv(os.path.join("Data_Set",'train_modified_data.csv'))
test_preprocessed= pd.read_csv(os.path.join("Data_Set",'test_modified_data.csv'))

def excel_doc():
    
    # Export the subset data to an Excel file [Only the first 10 rows] // might need to install Excel viewer extention in vs code 
    data.head(10).to_excel(os.path.join("Excel", 'table_of_data.xlsx'), index=False)  
    test_data.head(10).to_excel(os.path.join("Excel", 'table_of_test_data.xlsx'), index=False)
    train_preprocessed.head(10).to_excel(os.path.join("Excel", 'table_processed_train.xlsx'), index=False)
    test_preprocessed.head(10).to_excel(os.path.join("Excel", 'table_processed_test.xlsx'), index=False)

def vis_compound_RT():
    # Filter the data for the first 25 drugs
    first_25_drugs = data['Compound'].value_counts().index[:25]  # Get the names of the first 10 drugs
    filtered_data = data[data['Compound'].isin(first_25_drugs)]

    # Create the scatter plot (Compound vs RT : color coded with the Lab)
    sns.scatterplot(x='Compound', y='RT', hue='Lab', data=filtered_data)
    plt.xlabel('Compound')
    plt.ylabel('Retention Time (RT)')
    plt.title('Retention Time vs Compound by Lab (First 25 Drugs)')
    plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels if needed
    plt.legend(title='Lab', bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 5} )  # Adjust legend position
    plt.tight_layout()
    plt.savefig(os.path.join("visualisation", 'scatter_plot.jpg'))
    plt.show()

"""excel_doc()
print ("excel done")
pf.linear_model(train_preprocessed,test_preprocessed)
print("linear done")
pf.poisson_regression(train_preprocessed,test_preprocessed)
print ("poison done")
pf.knn_regression(train_preprocessed,test_preprocessed)
print ("knn done")"""

data= pd.read_csv('train.csv')
cddd = pd.read_csv('cddd.csv')
# Export the subset data to an Excel file // might need to install Excel viewer extention in vs code
data.head(10).to_excel('table_of_data.xlsx', index=False)  # Displaying the first 10 rows as an example

n =25

dV.scatterRTvsCompound(data, n)
dV.RTvsCDDD(dV.mergeRT_CDDD(data, cddd, 100))
dV.HeatMap(dV.mergeRT_CDDD(data, cddd))
dV.RTvsCompoundbyLab(data, n, save = True)