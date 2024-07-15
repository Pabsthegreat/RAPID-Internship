import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
from total_calorie_predictor import totalcal

# Example: Suppress warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

# Load your dataset
df = pd.read_csv('RAPID\\datasets\\nonfood\\filteredcalbreak.csv')

# Select relevant features and target variables
X = df[['age','height','weight','activity_level','total_daily_expenditure']]  # Include other relevant features like age, gender, etc.
y_fat = df['fat']
y_carbs = df['carbs']
y_protein = df['protein']

# Split the data into training and testing sets
X_train, X_test, y_fat_train, y_fat_test = train_test_split(X, y_fat, test_size=0.2, random_state=0)
X_train, X_test, y_carbs_train, y_carbs_test = train_test_split(X, y_carbs, test_size=0.2, random_state=0)
X_train, X_test, y_protein_train, y_protein_test = train_test_split(X, y_protein, test_size=0.2, random_state=0)

# Train Lasso regression models
lasso_fat = Lasso(alpha=40).fit(X_train, y_fat_train)
lasso_carbs = Lasso(alpha=40).fit(X_train, y_carbs_train)
lasso_protein = Lasso(alpha=40).fit(X_train, y_protein_train)

# Predict and evaluate the models
y_fat_pred = lasso_fat.predict(X_test)
y_carbs_pred = lasso_carbs.predict(X_test)
y_protein_pred = lasso_protein.predict(X_test)

# Example prediction usage
new_data = [totalcal()]  # Example with total daily expenditure, fat, carbs, protein
print(new_data)
predicted_fat = lasso_fat.predict(new_data)
predicted_carbs = lasso_carbs.predict(new_data)
predicted_protein = lasso_protein.predict(new_data)

print(f"Predicted Fat Calories: {predicted_fat[0]}")
print(f"Predicted Carbs Calories: {predicted_carbs[0]}")
print(f"Predicted Protein Calories: {predicted_protein[0]}")




# Calculate evaluation metrics for fat prediction model
# mse_fat = mean_squared_error(y_fat_test, y_fat_pred)
# r2_fat = r2_score(y_fat_test, y_fat_pred)

# print('Fat Prediction Model:')
# print('Mean Absolute Error:', mean_absolute_error(y_fat_test, y_fat_pred))
# print('Mean Squared Error:', mse_fat)
# print('R-squared:', r2_fat)
# print()

# Calculate evaluation metrics for carbs prediction model
# mse_carbs = mean_squared_error(y_carbs_test, y_carbs_pred)
# r2_carbs = r2_score(y_carbs_test, y_carbs_pred)

# print('Carbs Prediction Model:')
# print('Mean Absolute Error:', mean_absolute_error(y_carbs_test, y_carbs_pred))
# print('Mean Squared Error:', mse_carbs)
# print('R-squared:', r2_carbs)
# print()

# Calculate evaluation metrics for protein prediction model
# mse_protein = mean_squared_error(y_protein_test, y_protein_pred)
# r2_protein = r2_score(y_protein_test, y_protein_pred)

# print('Protein Prediction Model:')
# print('Mean Absolute Error:', mean_absolute_error(y_protein_test, y_protein_pred))
# print('Mean Squared Error:', mse_protein)
# print('R-squared:', r2_protein)
# print()