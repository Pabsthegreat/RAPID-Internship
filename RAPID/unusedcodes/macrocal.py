# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Hypothetical data
# data = {
#     'BMI': [25, 30, 22, 28, 35],
#     'Postprandial_Level': [150, 180, 140, 160, 190],
#     'Calories_Per_Day': [2000, 2500, 1800, 2200, 2700],
#     'Carbs_Proportion': [50, 55, 45, 50, 60],  # In percentage
#     'Protein_Proportion': [30, 25, 35, 30, 25],  # In percentage
#     'Fat_Proportion': [20, 20, 20, 20, 15]  # In percentage
# }

# df = pd.DataFrame(data)

# # Independent variables
# X = df[['BMI', 'Postprandial_Level']]

# # Dependent variables
# y_calories = df['Calories_Per_Day']
# y_carbs = df['Carbs_Proportion']
# y_protein = df['Protein_Proportion']
# y_fat = df['Fat_Proportion']

# # Split the data into training and testing sets
# X_train, X_test, y_calories_train, y_calories_test = train_test_split(X, y_calories, test_size=0.2, random_state=0)
# X_train, X_test, y_carbs_train, y_carbs_test = train_test_split(X, y_carbs, test_size=0.2, random_state=0)
# X_train, X_test, y_protein_train, y_protein_test = train_test_split(X, y_protein, test_size=0.2, random_state=0)
# X_train, X_test, y_fat_train, y_fat_test = train_test_split(X, y_fat, test_size=0.2, random_state=0)

# # Create and fit the models
# model_calories = LinearRegression()
# model_calories.fit(X_train, y_calories_train)

# model_carbs = LinearRegression()
# model_carbs.fit(X_train, y_carbs_train)

# model_protein = LinearRegression()
# model_protein.fit(X_train, y_protein_train)

# model_fat = LinearRegression()
# model_fat.fit(X_train, y_fat_train)

# # Predict and evaluate the models
# y_calories_pred = model_calories.predict(X_test)
# y_carbs_pred = model_carbs.predict(X_test)
# y_protein_pred = model_protein.predict(X_test)
# y_fat_pred = model_fat.predict(X_test)

# # Print evaluation metrics
# print('Calories Per Day Model:')
# print('Mean Absolute Error:', mean_absolute_error(y_calories_test, y_calories_pred))
# print('Mean Squared Error:', mean_squared_error(y_calories_test, y_calories_pred))
# print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_calories_test, y_calories_pred)))
# print('R-squared:', model_calories.score(X_test, y_calories_test))

# # print('\nCarbs Proportion Model:')
# # print('Mean Absolute Error:', mean_absolute_error(y_carbs_test, y_carbs_pred))
# # print('Mean Squared Error:', mean_squared_error(y_carbs_test, y_carbs_pred))
# # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_carbs_test, y_carbs_pred)))
# # print('R-squared:', model_carbs.score(X_test, y_carbs_test))

# # print('\nProtein Proportion Model:')
# # print('Mean Absolute Error:', mean_absolute_error(y_protein_test, y_protein_pred))
# # print('Mean Squared Error:', mean_squared_error(y_protein_test, y_protein_pred))
# # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_protein_test, y_protein_pred)))
# # print('R-squared:', model_protein.score(X_test, y_protein_test))

# # print('\nFat Proportion Model:')
# # print('Mean Absolute Error:', mean_absolute_error(y_fat_test, y_fat_pred))
# # print('Mean Squared Error:', mean_squared_error(y_fat_test, y_fat_pred))
# # print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_fat_test, y_fat_pred)))
# # print('R-squared:', model_fat.score(X_test, y_fat_test))

# # Example prediction for a new patient
# new_patient = pd.DataFrame({'BMI': [25.95], 'Postprandial_Level': [140]})
# predicted_calories = model_calories.predict(new_patient)
# predicted_carbs = model_carbs.predict(new_patient)
# predicted_protein = model_protein.predict(new_patient)
# predicted_fat = model_fat.predict(new_patient)

# print('\nPredicted Daily Calories:', predicted_calories)
# print('Predicted Carbs Proportion (%):', predicted_carbs)
# print('Predicted Protein Proportion (%):', predicted_protein)
# print('Predicted Fat Proportion (%):', predicted_fat)


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Load the dataset
# data = pd.read_csv('cleaned_dataset.csv')

# # Display the first few rows of the dataset
# print(data.head())

# # Calculate BMI
# data['BMI'] = data['weight'] / ((data['height'] / 100) ** 2)

# # Independent variables
# X = data[['BMI']]

# # Dependent variable
# y_calories = data['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_calories_train, y_calories_test = train_test_split(X, y_calories, test_size=0.2, random_state=0)

# # Create and fit the model
# model_calories = LinearRegression()
# model_calories.fit(X_train, y_calories_train)

# # Predict on the test set
# y_calories_pred = model_calories.predict(X_test)

# # Print evaluation metrics
# print('Calories Per Day Model:')
# print('Mean Absolute Error:', mean_absolute_error(y_calories_test, y_calories_pred))
# print('Mean Squared Error:', mean_squared_error(y_calories_test, y_calories_pred))
# print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_calories_test, y_calories_pred)))
# print('R-squared:', model_calories.score(X_test, y_calories_test))

# # Example prediction for a new patient
# new_patient_height = 170  # in cm
# new_patient_weight = 70   # in kg
# new_patient_bmi = new_patient_weight / ((new_patient_height / 100) ** 2)

# new_patient = pd.DataFrame({'BMI': [new_patient_bmi]})
# predicted_calories = model_calories.predict(new_patient)

# print('\nPredicted Daily Calories:', predicted_calories)

