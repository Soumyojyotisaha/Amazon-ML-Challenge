import pandas as pd

# Path to your test.csv file
test_csv_path = 'C:/Users/Soumyojyoti Saha/OneDrive - vit.ac.in/Desktop/amazon ml/student_resource 3/dataset/test.csv'

# Load the dataset
test_df = pd.read_csv(test_csv_path)

# Print the first few rows of the dataset
print(test_df.head())

# Print the column names to identify the ground truth column
print(test_df.columns)
