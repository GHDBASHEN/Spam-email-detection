import pandas as pd

# Load the dataset
data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Selecting relevant columns
data.columns = ['label', 'message']  # Renaming columns
print(data.head())
