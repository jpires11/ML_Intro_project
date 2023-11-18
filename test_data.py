
import pandas as pd

def test_missing_values():
    """testing the data for missing values
    """
    # Load your test dataset
    data = pd.read_csv('train.csv')  # Replace with your test dataset

    # Check for missing values
    missing_values = data.isnull().sum()

    # Assert that no missing values should be present in the dataset
    assert missing_values.sum() == 0, f"Missing values found: {missing_values}"