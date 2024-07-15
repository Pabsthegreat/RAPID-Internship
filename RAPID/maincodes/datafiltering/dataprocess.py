# import csv

# # Function to remove quotes and the last column from each row
# def process_csv(input_file):
#     output_data = []
#     with open(input_file, 'r', newline='') as csvfile:
#         reader = csv.reader(csvfile)
#         for row in reader:
#             # Remove quotes from each element
#             cleaned_row = [element.strip('"') for element in row[:-1]]  # Remove quotes and skip the last column
#             output_data.append(cleaned_row)
    
#     # Write the processed data back to the same file
#     with open(input_file, 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerows(output_data)

# # Replace 'your_file.csv' with your actual CSV file name
# input_file = 'db2.csv'

# # Process the CSV file
# process_csv(input_file)


import pandas as pd

# Load your dataset (replace 'your_dataset.csv' with your actual dataset path)
df = pd.read_csv('db3.csv')

# Define the columns to check for outliers
columns_to_check = ['age', 'height', 'weight', 'bmr', 'activity_level', 'total_daily_expenditure']

# Function to remove outliers using IQR method
def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

# Remove outliers from specified columns
df_cleaned = remove_outliers(df, columns_to_check)

# Optionally, you can save the cleaned dataset
df_cleaned.to_csv('cleaned_dataset.csv', index=False)

# Display or further process the cleaned dataset
print("Cleaned dataset shape:", df_cleaned.shape)

# Example: Display the first few rows of the cleaned dataset
print(df_cleaned.head())




