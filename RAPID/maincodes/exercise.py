def recommend_exercise(age, bmi, activity_level):
    age_group = None
    bmi_category = None
    exercise_suggestions = []

    # Determine age group
    if age >= 18 and age <= 30:
        age_group = "Youth"
    elif age >= 31 and age <= 50:
        age_group = "Adult"
    elif age > 50:
        age_group = "Senior"
    
    # Determine BMI category
    if bmi < 18.5:
        bmi_category = "Underweight"
    elif 18.5 <= bmi < 24.9:
        bmi_category = "Normal weight"
    elif 25 <= bmi < 29.9:
        bmi_category = "Overweight"
    else:
        bmi_category = "Obese"
    
    # Determine exercise recommendations based on activity level
    if activity_level == "Sedentary":
        if bmi_category == "Underweight":
            exercise_suggestions = ["Walking", "Yoga", "Beginner's calisthenics"]
        elif bmi_category == "Normal weight":
            exercise_suggestions = ["Walking", "Basic strength training", "Yoga"]
        elif bmi_category == "Overweight":
            exercise_suggestions = ["Walking", "Water aerobics", "Low-impact activities"]
        else:
            exercise_suggestions = ["Walking", "Water aerobics", "Chair exercises"]
    
    elif activity_level == "Lightly active":
        if bmi_category == "Underweight":
            exercise_suggestions = ["Walking", "Swimming", "Light strength training"]
        elif bmi_category == "Normal weight":
            exercise_suggestions = ["Running", "Swimming", "Bodyweight exercises"]
        elif bmi_category == "Overweight":
            exercise_suggestions = ["Swimming", "Cycling", "Light strength training"]
        else:
            exercise_suggestions = ["Swimming", "Cycling (stationary bike)", "Resistance training"]
    
    elif activity_level == "Moderately active":
        if bmi_category == "Underweight":
            exercise_suggestions = ["Running", "Cycling", "Moderate strength training"]
        elif bmi_category == "Normal weight":
            exercise_suggestions = ["Running", "Cycling", "Intermediate strength training"]
        elif bmi_category == "Overweight":
            exercise_suggestions = ["Running", "HIIT (low-impact)", "Moderate strength training"]
        else:
            exercise_suggestions = ["Water aerobics", "Walking", "Low-impact HIIT"]
    
    elif activity_level == "Very active":
        if bmi_category == "Underweight":
            exercise_suggestions = ["HIIT", "Advanced strength training"]
        elif bmi_category == "Normal weight":
            exercise_suggestions = ["HIIT", "Advanced strength training", "Competitive sports"]
        elif bmi_category == "Overweight":
            exercise_suggestions = ["Advanced strength training", "Running", "Cycling"]
        else:
            exercise_suggestions = ["Supervised strength training", "Swimming", "Stationary cycling"]
    
    return {
        "Age Group": age_group,
        "BMI Category": bmi_category,
        "Exercise Suggestions": exercise_suggestions
    }

# Example usage
age = 35
bmi = 26
activity_level = "Moderately active"

recommendation = recommend_exercise(age, bmi, activity_level)
print(recommendation)
