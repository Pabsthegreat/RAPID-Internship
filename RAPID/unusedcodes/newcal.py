# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler

# # Load the dataset
# data = pd.read_csv('db2.csv')
# print(data.columns)
# # Display the first few rows of the dataset
# print(data.head())
# # Check for missing values
# print(data.isnull().sum())

# # Drop or fill missing values
# data = data.dropna()

# # Feature scaling
# scaler = StandardScaler()
# numerical_features = ['height', 'weight', 'age', 'activity_level', 'preprandial_level']
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Define the features (X) and target (y)
# X = data[['height', 'weight', 'age', 'activity_level', 'preprandial_level']]
# y = data['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')

# # BMR calculation function
# def calculate_bmr(weight, height, age, gender):
#     # Height is assumed to be in centimeters
#     if gender == 'male':
#         bmr = 10 * weight + 6.25 * height - 5 * age + 5
#     else:
#         bmr = 10 * weight + 6.25 * height - 5 * age - 161
#     return bmr

# # Calorie breakdown function
# def calculate_calorie_breakdown(row):
#     bmr = calculate_bmr(row['weight'], row['height'], row['age'], row['gender'])
#     activity_calories = bmr * row['activity_level']
#     tef = 0.1 * (bmr + activity_calories)  # TEF is roughly 10% of the total calories
#     return {'bmr': bmr, 'activity_calories': activity_calories, 'tef': tef}

# # Apply the function to each row in the dataset
# data[['bmr', 'activity_calories', 'tef']] = data.apply(calculate_calorie_breakdown, axis=1, result_type='expand')

# # Display the updated dataset with calorie breakdown
# print(data.head())

# # Add predictions to the test set for visualization
# X_test['predicted_total_daily_expenditure'] = y_pred

# # Add calorie breakdown to the test set
# def add_calorie_breakdown(row):
#     breakdown = calculate_calorie_breakdown(row)
#     return pd.Series(breakdown)

# calorie_breakdown = X_test.apply(add_calorie_breakdown, axis=1)
# X_test = pd.concat([X_test, calorie_breakdown], axis=1)

# # Display the test set with predictions and calorie breakdown
# print(X_test.head())




# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Load the dataset
# data = pd.read_csv('db2.csv')

# # Display the first few rows of the dataset
# print(data.head())

# # Check for missing values
# print(data.isnull().sum())

# # Drop or fill missing values
# data = data.dropna()

# # Encode categorical variable 'gender'
# label_encoder = LabelEncoder()
# data['gender'] = label_encoder.fit_transform(data['gender'])

# # Reorganize columns if 'gender' is the second column
# cols = data.columns.tolist()
# cols = cols[:1] + cols[2:] + [cols[1]]
# data = data[cols]

# # Feature scaling for numerical features
# scaler = StandardScaler()
# numerical_features = ['height', 'weight', 'age', 'activity_level', 'preprandial_level']
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Define the features (X) and target (y)
# X = data[['height', 'weight', 'age', 'activity_level', 'preprandial_level', 'gender']]
# y = data['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')




# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Load the dataset
# data = pd.read_csv('db2.csv')

# # Drop or fill missing values
# data = data.dropna()

# # Encode categorical variable 'gender'
# label_encoder = LabelEncoder()
# data['gender'] = label_encoder.fit_transform(data['gender'])

# # Feature engineering: Compute BMI
# data['bmi'] = data['weight'] / (data['height'] / 100) ** 2

# # Reorganize columns if 'gender' is the second column
# cols = data.columns.tolist()
# cols = cols[:1] + cols[2:] + [cols[1]]
# data = data[cols]

# # Feature scaling for numerical features
# scaler = StandardScaler()
# numerical_features = ['height', 'weight', 'age', 'activity_level', 'preprandial_level', 'bmi']
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Define the features (X) and target (y)
# X = data[['height', 'weight', 'age', 'activity_level', 'preprandial_level', 'bmi', 'gender']]
# y = data['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the Ridge Regression model
# model = Ridge(alpha=1.0)  # Adjust alpha for regularization strength
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')

# # BMR calculation function
# def calculate_bmr(weight, height, age, gender):
#     # Height is assumed to be in centimeters
#     if gender == 1:  # 1 represents 'male' after encoding
#         bmr = 10 * weight + 6.25 * height*100 - 5 * age + 5
#     else:
#         bmr = 10 * weight + 6.25 * height*100 - 5 * age - 161
#     return bmr

# # Calorie breakdown function
# def calculate_calorie_breakdown(row):
#     bmr = calculate_bmr(row['weight'], row['height'], row['age'], row['gender'])
#     activity_calories = bmr * row['activity_level']
#     tef = 0.1 * (bmr + activity_calories)  # TEF is roughly 10% of the total calories
#     return pd.Series({'bmr': bmr, 'activity_calories': activity_calories, 'tef': tef})

# # Apply the function to each row in the dataset
# data[['bmr', 'activity_calories', 'tef']] = data.apply(calculate_calorie_breakdown, axis=1)

# # Display the updated dataset with calorie breakdown
# print(data.head())

# # Add predictions to the test set for visualization
# X_test['predicted_total_daily_expenditure'] = y_pred

# # Add calorie breakdown to the test set
# calorie_breakdown = X_test.apply(calculate_calorie_breakdown, axis=1)
# X_test = pd.concat([X_test, calorie_breakdown], axis=1)

# # Display the test set with predictions and calorie breakdown
# print(X_test.head())



# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer

# # Load the dataset (adjust 'db2.csv' to your actual dataset path)
# data = pd.read_csv('db2.csv')

# # Check for missing values
# print(data.isnull().sum())

# # Drop or fill missing values
# data = data.dropna()

# # Feature scaling and encoding
# scaler = StandardScaler()
# numerical_features = ['height', 'weight', 'age', 'activity_level', 'preprandial_level']
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Encode categorical variable 'gender' to numerical values
# # Assuming 'gender' is the second column (index 1)
# ct = ColumnTransformer(
#     [('encoder', OneHotEncoder(), [1])],  # Encode 'gender' column
#     remainder='passthrough'
# )
# X_encoded = ct.fit_transform(data)

# # Define the features (X) and target (y)
# X = X_encoded[:, :-1]  # Exclude the last column (total_daily_expenditure)
# y = data['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Function to predict calories based on user input
# def predict_calories(height, weight, age, activity_level, gender):
#     # Prepare input data in the same format as the model expects
#     # Scale numerical features
#     user_data_scaled = scaler.transform([[height, weight, age, activity_level, 0]])  # Dummy gender (0) for scaling
#     # Encode categorical features
#     user_data_encoded = ct.transform(user_data_scaled)
    
#     # Predict total daily expenditure using the trained model
#     predicted_calories = model.predict(user_data_encoded)
#     return predicted_calories[0]

# # Example usage:
# height = float(input("Enter height in cm: "))
# weight = float(input("Enter weight in kg: "))
# age = int(input("Enter age in years: "))
# activity_level = float(input("Enter activity level (1.2 for sedentary to 1.9 for very active): "))
# gender = int(input("Enter gender (0 for female, 1 for male): "))

# predicted_calories = predict_calories(height, weight, age, activity_level, gender)
# print(f"Predicted total daily calorie expenditure: {predicted_calories:.2f} calories")


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler

# # Load the dataset
# data = pd.read_csv('cleaned_dataset.csv')

# # Check for missing values
# print(data.isnull().sum())

# # Drop or fill missing values
# data = data.dropna()

# # Define the features (X) and target (y)
# X = data[['bmr']]
# y = data['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')

# # Display coefficients
# coefficients = pd.DataFrame(model.coef_, columns=['Coefficient'], index=X.columns)
# print("Coefficients:")
# print(coefficients)

# # Function to predict total daily expenditure based on BMR
# def predict_total_daily_expenditure(bmr):
#     predicted_calories = model.predict([[bmr]])
#     return predicted_calories[0]

# # Example usage
# bmr_input = float(input("Enter BMR value: "))
# predicted_calories = predict_total_daily_expenditure(bmr_input)
# print(f"Predicted Total Daily Expenditure: {predicted_calories} calories")










# Further functionality (e.g., BMR calculation, calorie breakdown) can be added similarly as before




# # Further functionality (e.g., BMR calculation, calorie breakdown) can be added similarly as before


























# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler

# # Function to calculate BMR
# def calculate_bmr(weight, height, age, gender):
#     if gender == 1:  # Male
#         return 10 * weight + 6.25 * height - 5 * age + 5
#     elif gender == 0:  # Female
#         return 10 * weight + 6.25 * height - 5 * age - 161
#     else:
#         raise ValueError("Gender must be 1 (male) or 0 (female)")

# # Load the dataset
# data = pd.read_csv('db3.csv')

# # Display the first few rows of the dataset and verify column names
# print(data.columns)
# print(data.head())

# # Check for missing values
# print(data.isnull().sum())

# # Drop or fill missing values
# data = data.dropna()

# # Replace gender with numerical values
# data['gender'] = data['gender'].apply(lambda x: 1 if x == 'male' else 0)

# # Calculate BMR for each row
# data['bmr'] = data.apply(lambda row: calculate_bmr(row['weight'], row['height'], row['age'], row['gender']), axis=1)

# # Feature scaling for numerical features
# scaler = StandardScaler()
# numerical_features = ['height', 'weight', 'age', 'activity_level', 'bmr']
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Define the features (X) and target (y)
# X = data[['height', 'weight', 'age', 'activity_level', 'bmr', 'gender']]
# y = data['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the multiple linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')

# # Print the coefficients and intercept
# print(f'Coefficients: {model.coef_}')
# print(f'Intercept: {model.intercept_}')

# # Plotting actual vs. predicted values
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', label='Perfect Prediction')
# plt.title('Actual vs. Predicted Total Daily Expenditure')
# plt.xlabel('Actual Total Daily Expenditure')
# plt.ylabel('Predicted Total Daily Expenditure')
# plt.legend()
# plt.grid(True)
# plt.show()

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# # Function to calculate BMR
# def calculate_bmr(weight, height, age, gender):
#     if gender == 1:  # Male
#         return 10 * weight + 6.25 * height - 5 * age + 5
#     elif gender == 0:  # Female
#         return 10 * weight + 6.25 * height - 5 * age - 161
#     else:
#         raise ValueError("Gender must be 1 (male) or 0 (female)")

# # Load the dataset (assuming db2.csv contains similar data used for training)
# data = pd.read_csv('db2.csv')

# # Display the first few rows of the dataset and verify column names
# print(data.columns)
# print(data.head())

# # Check for missing values
# print(data.isnull().sum())

# # Drop or fill missing values
# data = data.dropna()

# # Replace gender with numerical values
# data['gender'] = data['gender'].apply(lambda x: 1 if x == 'male' else 0)

# # Calculate BMR for each row
# data['bmr'] = data.apply(lambda row: calculate_bmr(row['weight'], row['height'], row['age'], row['gender']), axis=1)

# # Feature scaling for numerical features
# scaler = StandardScaler()
# numerical_features = ['height', 'weight', 'age', 'activity_level', 'bmr', 'gender']  # Include 'gender' in numerical features
# data[numerical_features] = scaler.fit_transform(data[numerical_features])

# # Define the features (X) and target (y)
# X = data[['height', 'weight', 'age', 'activity_level', 'bmr', 'gender']]
# y = data['total_daily_expenditure']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the multiple linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = model.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')

# # Function to preprocess user input and predict
# def predict_total_daily_expenditure(height, weight, age, activity_level, gender):
#     # Calculate BMR based on user input
#     bmr = calculate_bmr(weight, height, age, gender)
    
#     # Scale input features
#     scaled_data = scaler.transform([[height, weight, age, activity_level, bmr, gender]])
    
#     # Predict total_daily_expenditure
#     total_daily_expenditure_pred = model.predict(scaled_data)
    
#     return total_daily_expenditure_pred[0]

# # Example usage: Interactive input from the user
# print("\nEnter parameters to predict total daily expenditure:")
# height = float(input("Height (cm): "))
# weight = float(input("Weight (kg): "))
# age = int(input("Age (years): "))
# activity_level = float(input("Activity Level (1-5): "))  # Example: 1.2
# gender = int(input("Gender (1 for male, 0 for female): "))  # Example: 1 or 0

# # Make prediction
# predicted_expenditure = predict_total_daily_expenditure(height, weight, age, activity_level, gender)

# print(f'\nPredicted Total Daily Expenditure: {predicted_expenditure:.2f} calories')






#ridge

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_error, r2_score

# # Load your dataset (replace 'your_dataset.csv' with your actual dataset path)
# df = pd.read_csv('cleaned_dataset.csv')

# # Selecting features and target variable
# X = df[['height', 'activity_level', 'weight','age']]
# y = df['total_daily_expenditure']

# # Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Initialize Ridge regression model
# ridge_model = Ridge(alpha=1.0)  # Adjust alpha parameter for regularization strength

# # Fit the model
# ridge_model.fit(X_train_scaled, y_train)

# # Make predictions
# y_pred = ridge_model.predict(X_test_scaled)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f'Mean Squared Error: {mse}')
# print(f'R^2 Score: {r2}')

# # Optional: Print coefficients if needed
# print("Ridge Coefficients:")
# print(ridge_model.coef_)