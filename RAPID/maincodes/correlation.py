#correlation analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load your dataset (replace 'your_dataset.csv' with your actual dataset path)
df = pd.read_csv('filteredcalbreak.csv')

# Encode 'gender' column using LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])

# Compute correlation matrix
corr_matrix = df.corr()

# Plotting the correlation matrix using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()


#feature importance
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
df = pd.read_csv('filteredcalbreak.csv')

# Replace 'gender' values with 0 and 1
df['gender'] = df['gender'].replace({'M': 0, 'F': 1})

# Assuming X and y are defined as features and target variable respectively
X = df[['age', 'gender', 'height', 'weight', 'activity_level', 'carb','protein','fat']]
y = df['total_daily_expenditure']

# Initialize Random Forest model
rf_model = RandomForestRegressor()

# Fit the model
rf_model.fit(X, y)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the sorted feature importances
print(feature_importance_df)


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import Lasso
# from sklearn.metrics import mean_squared_error
# from sklearn.preprocessing import StandardScaler

# # Load your dataset (replace 'your_dataset.csv' with your actual dataset path)
# df = pd.read_csv('calbreak.csv')

# # Convert 'gender' column to numerical values if needed
# df['gender'] = df['gender'].map({'M': 0, 'F': 1})

# # Define features and target variable
# features = ['height', 'weight', 'age', 'activity_level']
# X = df[features]
# y = df['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Define a range of alpha values to test
# alphas = np.logspace(-4, 2, 50)

# # Set up the Lasso regression model with GridSearchCV
# lasso = Lasso()
# param_grid = {'alpha': alphas}
# grid_search = GridSearchCV(lasso, param_grid, cv=5, scoring='neg_mean_squared_error')

# # Fit the model
# grid_search.fit(X_train_scaled, y_train)

# # Get the best alpha
# best_alpha = grid_search.best_params_['alpha']
# print(f"Best alpha: {best_alpha}")

# # Train the final model with the best alpha
# lasso_best = Lasso(alpha=best_alpha)
# lasso_best.fit(X_train_scaled, y_train)

# # Predict and evaluate the model
# y_pred = lasso_best.predict(X_test_scaled)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
# print(f"R^2 Score: {lasso_best.score(X_test_scaled, y_test)}")

# # Optional: Print coefficients
# print("Lasso Coefficients:")
# print(lasso_best.coef_)
