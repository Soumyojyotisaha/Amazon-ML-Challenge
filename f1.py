import pandas as pd
from sklearn.metrics import f1_score
import re

def normalize_value(value):
    """
    Normalize the value to a consistent format.
    """
    if pd.isna(value) or not value.strip():
        return ""
    
    # Remove extra spaces and convert to lower case
    value = value.strip().lower()
    
    # Extract number and unit
    match = re.match(r"(\d+(\.\d+)?)\s*([a-zA-Z]+)", value)
    
    if match:
        number, _, unit = match.groups()
        return f"{number} {unit}"
    
    return value  # Return original if not matching pattern

def calculate_f1_score(file_path):
    """
    Calculate the F1 score based on predictions and ground truth.
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Normalize values
    df['ground_truth'] = df['ground_truth'].apply(normalize_value)
    df['prediction'] = df['prediction'].apply(normalize_value)
    
    # Get true labels and predicted labels
    true_labels = df['ground_truth']
    pred_labels = df['prediction']
    
    # Calculate F1 score
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    return f1

if __name__ == "__main__":
    # Path to the CSV file containing predictions and ground truth
    file_path = 'dataset/combined_results.csv'
    
    # Calculate F1 score
    f1 = calculate_f1_score(file_path)
    
    # Print the F1 score
    print(f"F1 Score: {f1:.4f}")
