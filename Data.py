
import dataVis as dV
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data= pd.read_csv('train.csv')
cddd = pd.read_csv('cddd.csv')
# Export the subset data to an Excel file // might need to install Excel viewer extention in vs code
data.head(10).to_excel('table_of_data.xlsx', index=False)  # Displaying the first 10 rows as an example

n =25

dV.scatterRTvsCompound(data, n)
dV.RTvsCDDD(dV.mergeRT_CDDD(data, cddd, 100))
dV.HeatMap(dV.mergeRT_CDDD(data, cddd))
dV.RTvsCompoundbyLab(data, n, save = True)