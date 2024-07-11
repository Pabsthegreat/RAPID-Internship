import os
import pandas as pd

# Define the root directory where your data is stored
root_dir = 'RAPID\\datasets\\food'

def check_csv_file(file_path):
    """
    Check a single CSV file for consistency in the number of fields.
    """
    try:
        data = pd.read_csv(file_path,quotechar='"', delimiter=',', encoding='utf-8')
        expected_columns = data.shape[1]
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if len(line.split(',')) != expected_columns:
                    print(f"Inconsistent number of fields in file {file_path}, line {i + 1}")
    except pd.errors.ParserError as e:
        print(f"Parser error in file {file_path}: {e}")
    except Exception as e:
        print(f"An error occurred with file {file_path}: {e}")

def check_all_csv_files(root_dir):
    """
    Iterate through all regions, subregions, and meal time files to check for consistency.
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
                            check_csv_file(file_path)

# Run the check on all CSV files
check_all_csv_files(root_dir)


# import os
# import pandas as pd

# # Function to read a CSV file with proper handling of quoted strings and removing the MealTimings column
# def read_and_clean_csv_file(file_path):
#     try:
#         # Read the CSV file with proper quoting and handle missing values
#         df = pd.read_csv(file_path, quotechar='"', delimiter=',', keep_default_na=False)
#         # Remove the MealTimings column if it exists
#         if 'MealTimings' in df.columns:
#             df = df.drop(columns=['MealTimings'])
#         return df
#     except pd.errors.ParserError as e:
#         print(f"Error parsing {file_path}: {e}")
#         return None

# # Function to iterate through all regions, subregions, and files
# def process_all_files(base_path):
#     for region in os.listdir(base_path):
#         region_path = os.path.join(base_path, region)
#         if os.path.isdir(region_path):
#             for subregion in os.listdir(region_path):
#                 subregion_path = os.path.join(region_path, subregion)
#                 if os.path.isdir(subregion_path):
#                     for file_name in os.listdir(subregion_path):
#                         file_path = os.path.join(subregion_path, file_name)
#                         if file_path.endswith('.csv'):
#                             df = read_and_clean_csv_file(file_path)
#                             if df is not None:
#                                 # Save the modified dataframe back to the file
#                                 df.to_csv(file_path, index=False)

# # Define the base path to your dataset
# base_path = 'RAPID\\datasets\\food'
# process_all_files(base_path)
