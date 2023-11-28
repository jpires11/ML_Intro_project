import pandas as pd
import numpy as np
import prediction_functions as pf
import prepocessing as pre
import seaborn as sns
import matplotlib.pyplot as plt
import os #to save plots
#from rdkit import Chem
from openpyxl.workbook import Workbook


def scatterRTvsCompound(data, n, save = False):
    '''
    Creates a scatter plot of the compound vs the retention time.
    train.csv file is provided as a dataframe
    n is the number of compounds we want to have in the plot
    '''
    
    # Filter the data for the first 25 drugs
    first_drugs = data['Compound'].value_counts().index[:n]  # Get the names of the first 10 drugs
    filtered_data = data[data['Compound'].isin(first_drugs)]

    # Create the scatter plot (Compound vs RT : color coded with the Lab)
    sns.scatterplot(x='Compound', y='RT', hue='Lab', data=filtered_data)
    plt.xlabel('Compound')
    plt.ylabel('Retention Time (RT)')
    plt.title('Retention Time vs Compound by Lab (First ' + str(n) + ' Drugs)')
    plt.xticks(rotation=90, fontsize=8)  # Rotate x-axis labels if needed
    plt.legend(title='Lab', bbox_to_anchor=(1.05, 1), loc='upper left', prop={'size': 5} )  # Adjust legend position
    plt.tight_layout()
    if (save == True):
        plt.savefig(os.path.join("visualisation", 'scatter_plot.jpg'))
    plt.show()

def mergeRT_CDDD(data, cddd, n=512):
    #Assuming 'data' is your DataFrame
    merged_data = pd.merge(data, cddd, on='SMILES')
    # Select columns of interest
    selected_columns = ['RT'] + [f'cddd_{i}' for i in range(1, n)]
    subset_data = merged_data[selected_columns]
    return subset_data
def RTvsCDDD(subset_data, save = False):
    '''
    subset_data is a dataframe with RT and cddd only (obtainable with the mergeRT_CDDD function)
    '''

    # Melt the DataFrame to have cddd values in one column
    melted_data = pd.melt(subset_data, id_vars=['RT'], var_name='cddd', value_name='cddd_Value')

    # Create a scatter plot using Seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='cddd', y='RT', hue='cddd_Value', data=melted_data, palette='viridis', s=4)
    plt.title('2D HeatMap Plot of Retention Times with cddd Value as Color')
    plt.xlabel('cddd Column')
    plt.ylabel('RT')
    if (save == True):
        plt.savefig(os.path.join("visualisation", 'RTwithCDDDasColor.jpg'))
    plt.show()


def HeatMap(subset_data, save = False):
    '''
    subset_data is a dataframe with RT and cddd only (obtainable with the mergeRT_CDDD function)
    '''
    heatmap_data = subset_data.values[:, 1:]  # Exclude RT column

    # Create a heatmap using Matplotlib's imshow
    plt.figure(figsize=(12, 8))
    plt.imshow(heatmap_data, aspect='auto', cmap='viridis')

    # Set axis labels
    plt.xlabel('cddd Column')
    plt.ylabel('Sample Index')

    # Show colorbar
    plt.colorbar(label='RT')

    plt.title('Heatmap of cddd values with RT as Color')
    if (save == True):
        plt.savefig(os.path.join("visualisation", 'HeatmapCDDD_RT.jpg'))
    plt.show()

def RTvsCompoundbyLab(data, n, save = False):

    first_drugs = data['Compound'].value_counts().index[:n]  # Get the names of the first 10 drugs
    filtered_data = data[data['Compound'].isin(first_drugs)]
    sorted_df = filtered_data.sort_values(by='RT')
    labs = sorted_df['Lab'].unique().tolist()
    print(len(labs))

    fig, axes = plt.subplots(nrows=4, ncols=6, figsize=(20, 10), gridspec_kw={'wspace': .5, 'hspace': .5})
    axes = axes.flatten()


    # Flatten the axes array for easier iteration
    axes = axes.flatten()

    # Loop through each unique lab and plot using Seaborn
    for i, lab in enumerate(labs):
        if i < 24:  # Ensure that we don't exceed the number of subplots
            ax = axes[i]
            labdata = sorted_df[sorted_df['Lab'] == lab]
            sns.scatterplot(x='Compound', y='RT', data=labdata, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize= 3)
            ax.set_title(f'Lab {lab}', fontsize=6)
        else:
            break
    
    if (save == True):
        plt.savefig(os.path.join("visualisation", 'RTvsCompoundbyLab.jpg'))
    
    plt.show()

def GD_parameters(train_clean, test_data, save = False):
    X_,y_train,X_te= pre.create_sets(train_clean,test_data)
    X_train, X_val, y_train, y_val = pf.train_test_split(train_clean, y_train, test_size=0.2, random_state=42)
    train = pd.merge(X_train, y_train)

    # Initialize a range of learning rates to try
    learning_rates = [0.05, 0.01, 0.005]
    epochs = [300, 400, 500, 600, 700]
    # Dictionary to store results for each learning rate
    results = {}
    X_val = X_val.drop(['RT'], axis=1)


    for lr in learning_rates:
        # Train the model with the current learning rate
        y_pred = pf.gradient_descent(train, X_val, lr, epochs=500)  # Adjust epochs as needed
       
        # Ensure that y_val has the correct shape
        print(f"Shape of y_val: {X_val.shape}")

        # Evaluate the model using mean squared error
        mse = pf.mean_squared_error(y_val, y_pred)
        
        # Store the results for later analysis
        results[lr] = mse
    # Plot the learning rate vs. mean squared error
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xscale('log')  # Use a log scale for better visualization
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Rate Tuning')
    if (save == True):
        plt.savefig(os.path.join("visualisation", 'GradientDescentLRParameters.jpg'))
    plt.show()

    results = {}
    for ep in epochs:
        # Train the model with the current learning rate
        y_pred = pf.gradient_descent(train, X_val, 0.01, epochs=ep)  # Adjust epochs as needed
        
        # Ensure that y_val has the correct shape
        print(f"Shape of y_val: {X_val.shape}")

        # Evaluate the model using mean squared error
        mse = pf.mean_squared_error(y_val, y_pred)
        
        # Store the results for later analysis
        results[ep] = mse

    # Plot the epoch vs. mean squared error
    plt.plot(list(results.keys()), list(results.values()), marker='o')
    plt.xscale('log')  # Use a log scale for better visualization
    plt.xlabel('epochs')
    plt.ylabel('Mean Squared Error')
    plt.title('Epoch Tuning')
    if (save == True):
        plt.savefig(os.path.join("visualisation", 'GradientDescentEPParameters.jpg'))
    plt.show()