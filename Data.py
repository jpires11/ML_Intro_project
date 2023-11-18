
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os #to save plots
from rdkit import Chem
from openpyxl.workbook import Workbook
data= pd.read_csv('train.csv')

# Export the subset data to an Excel file // might need to install Excel viewer extention in vs code
data.head(10).to_excel('table_of_data.xlsx', index=False)  # Displaying the first 10 rows as an example

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

#visualisation with mol
"""
sns.scatterplot(x='mol', y='RT', hue='Compound', data=filtered_data)
plt.xlabel('Mol')
plt.ylabel('Retention Time (RT)')
plt.title('Retention Time vs Mol with Compound Variation (First 25 Drugs)')
plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels if needed
plt.legend(title='SMILES', bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 5})  # Adjust legend
plt.tight_layout()
"""
