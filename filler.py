import os
import pandas as pd

# Define the root directory where your data is stored
root_dir = 'RAPID\\datasets\\food'

def process_csv_file(file_path):
    """
    Process a single CSV file to add the rating column before meat_type column.
    If both side and needside are 0, set the rating to 5.
    """
    try:
        data = pd.read_csv(file_path)
        
        # Insert the rating column before the meat_type column
        meat_type_index = data.columns.get_loc('meat_type')
        data.insert(meat_type_index, 'rating', 0)
        
        # Set rating to 5 if both side and needside are 0
        data.loc[(data['Side'] == 0) & (data['NeedSide'] == 0), 'rating'] = 5
        
        # Save the modified DataFrame back to CSV
        data.to_csv(file_path, index=False)
        print(f"Processed file: {file_path}")
    except Exception as e:
        print(f"An error occurred with file {file_path}: {e}")

def process_all_csv_files(root_dir):
    """
    Iterate through all regions, subregions, and meal time files to process them.
    """
    for region in os.listdir(root_dir):
        region_path = os.path.join(root_dir, region)
        if os.path.isdir(region_path):
            for subregion in os.listdir(region_path):
                subregion_path = os.path.join(region_path, subregion)
                if os.path.isdir(subregion_path):
                    for meal_time_file in os.listdir(subregion_path):
                        if meal_time_file.endswith('.csv'):
                            file_path = os.path.join(subregion_path, meal_time_file)
                            process_csv_file(file_path)

# Run the process on all CSV files
process_all_csv_files(root_dir)
