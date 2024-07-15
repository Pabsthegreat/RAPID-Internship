import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# Set backend for non-interactive plotting
matplotlib.use('Agg')

# Load your dataset (replace 'calbreak.csv' with your actual dataset path)
df = pd.read_csv('calbreak.csv')

# Convert 'gender' column to numeric
df['gender'] = df['gender'].map({'M': 1, 'F': 0})

# Ensure there are no missing values in the specified columns
columns_to_check = ['carb', 'protein', 'fat']
df[columns_to_check] = df[columns_to_check].apply(pd.to_numeric, errors='coerce')
df.dropna(subset=columns_to_check, inplace=True)

# Function to remove outliers using the IQR method
def remove_outliers(df, columns, multiplier=1.5):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return df[~((df[columns] < lower_bound) | (df[columns] > upper_bound)).any(axis=1)]

# Adjusting the IQR multiplier to be less strict
df_cleaned = remove_outliers(df, columns_to_check, multiplier=1.5)

# Optionally, you can save the cleaned dataset
df_cleaned.to_csv('new.csv', index=False)

# Display or further process the cleaned dataset
print("Cleaned dataset shape:", df_cleaned.shape)
print(df_cleaned.head())

# Plotting the distribution of each column
# for column in columns_to_check:
#     plt.figure(figsize=(10, 4))
#     sns.boxplot(x=df_cleaned[column])
#     plt.title(f'Distribution of {column}')
#     plt.savefig(f'{column}_distribution.png')
#     plt.close()
