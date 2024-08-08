#lasso
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score

def totalcal():
    height = float(input("Enter your height in cm: "))
    weight = float(input("Enter your weight in kg: "))
    age = int(input("Enter your age in years: "))
    activity_level = float(input("Enter your activity level (1.2 for sedentary, 1.375 for lightly active, 1.55 for moderately active, 1.725 for very active, 1.9 for extra active): "))
    df = pd.read_csv("RAPID\\datasets\\nonfood\\cleaned_dataset.csv")

    # Selecting features and target variable
    features_input = ['height', 'weight', 'age', 'activity_level']
    target_variable = 'total_daily_expenditure'

    X = df[features_input]
    y = df[target_variable]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize Lasso regression model
    alpha = 30  
    lasso_model = Lasso(alpha=alpha, random_state=42)

    # Fit the model
    lasso_model.fit(X_train, y_train)

    # Make prediction for user input
    user_data = pd.DataFrame([[height, weight, age, activity_level]], columns=features_input)
    predicted_calories = int(lasso_model.predict(user_data))
    print("Total daily expenditure:",predicted_calories)

    return [age,height,weight,activity_level,predicted_calories]
