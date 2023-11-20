
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os #to save plots
from openpyxl.workbook import Workbook
#load data
data= pd.read_csv('train.csv')
test_data=pd.read_csv("test.csv")

def excel_doc():
    
    # Export the subset data to an Excel file // might need to install Excel viewer extention in vs code
    data.head(10).to_excel('table_of_data.xlsx', index=False)  # Displaying the first 10 rows as an example
    test_data.head(10).to_excel('table_of_test_data.xlsx', index=False)
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
    
def linear_model():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error
    import numpy as np
    
    X = data[[f'ECFP_{i}' for i in range(1, 1025)]]  # Adjust column names accordingly
    y = data['RT']
    # Split the data into training and testing sets
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = data[[f'ECFP_{i}' for i in range(1, 1025)]]  # Adjust column names accordingly
    y_train = data['RT']
    X_test = test_data[[f'ECFP_{i}' for i in range(1, 1025)]]  # Adjust column names accordingly
    
    # Initialize and train the linear regression model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = linear_model.predict(X_test)
    
    
    # Save the predictions to a CSV file
    output_df = pd.DataFrame({'Predicted_RT': y_pred})
    output_df.to_csv('predicttion_linear_model.csv', index=False) 
    # Calculate RMSE
    #test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    #print(f"Test RMSE: {test_rmse:.4f}")
    
linear_model()