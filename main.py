import os
import pandas as pd
from src.utils import download_images
from src.constants import entity_unit_map
from src.sanity import sanity_check
import random

def predictor(image_link, category_id, entity_name):
    '''
    Call your model/approach here to generate predictions
    '''
    # Download the image (if necessary)
    image_path = 'temp_image.jpg'
    download_images([image_link], 'temp_images', allow_multiprocessing=False)
    
    # Use the entity_name to fetch allowed units
    allowed_units = entity_unit_map.get(entity_name, [])
    if random.random() > 0.5 and allowed_units:
        unit = random.choice(list(allowed_units))
        return f"{random.uniform(1, 100):.2f} {unit}"
    return ""  # Failed prediction

def calculate_f1_score(test_df):
    '''
    Calculate the F1 score based on the ground truth and predictions
    '''
    # Initialize counts
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    for _, row in test_df.iterrows():
        gt = row['ground_truth']  # Replace 'ground_truth' with the actual column name for ground truth
        pred = row['prediction']  # Prediction
        
        if pred != "" and gt != "" and pred == gt:
            true_positives += 1
        elif pred != "" and gt != "" and pred != gt:
            false_positives += 1
        elif pred != "" and gt == "":
            false_positives += 1
        elif pred == "" and gt != "":
            false_negatives += 1
        elif pred == "" and gt == "":
            true_negatives += 1
    
    # Calculate Precision and Recall
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Calculate F1 Score
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1_score

def main():
    # Adjust the dataset folder path
    DATASET_FOLDER = 'dataset/'  # Ensure this matches your directory structure
    
    # Load test data
    test_csv_path = os.path.join(DATASET_FOLDER, 'test.csv')
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV file not found at {test_csv_path}")
    test = pd.read_csv(test_csv_path)
    
    # Print column names to debug
    print("Column names in test.csv:")
    print(test.columns)
    
    # Ensure the temp_images directory exists
    os.makedirs('temp_images', exist_ok=True)
    
    # Generate predictions
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    # Debugging: Print out the predictions to check if they are being generated
    print("Generated predictions:")
    print(test[['index', 'prediction']])
    
    # Save successful predictions to test_out.csv
    test_out = test[test['prediction'] != ""]
    output_filename_out = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test_out[['index', 'prediction']].to_csv(output_filename_out, index=False)
    
    # Verify that test_out.csv is saved in the correct CSV format
    test_out_check = pd.read_csv(output_filename_out)
    print("Preview of test_out.csv:")
    print(test_out_check.head())
    
    # Save failed predictions to test_fail.csv
    test_fail = test[test['prediction'] == ""]
    output_filename_fail = os.path.join(DATASET_FOLDER, 'test_fail.csv')
    test_fail[['index', 'prediction']].to_csv(output_filename_fail, index=False)
    
    # Perform sanity check
    sanity_check(test_csv_path, output_filename_out)
    
    # Calculate and print F1 score
    f1_score = calculate_f1_score(test)
    print(f"F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    main()
