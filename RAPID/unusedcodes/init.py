import json
from flask import Flask, request, render_template_string,jsonify
import csv
import random
from pathlib import Path
import google.generativeai as genai
api_key = 'AIzaSyCQtfB9Yk14EGRS2FzQ79YF_4Q9J6-eZt4'
genai.configure(api_key=api_key)
app = Flask(__name__)

@app.route('/getTargetedCalories', methods=['POST'])
def getTargetedCalories():
    
    if request.method == 'POST':
        # Storing the form data in variables
        data = request.get_json()
    

        # Updated variable assignments using data.get()
        age = int(data['age']) if data.get('age', '0').isdigit() else 0
        height = int(data['height']) if data.get('height', '0').isdigit() else 0
        weight = int(data['weight']) if data.get('weight', '0').isdigit() else 0
        sex = data.get('sex', '')
        workout_goal = data.get('workout_goal', '')
        workout_split = data.get('workout_split', '')
        user_level = data.get('user_level', '')
        region = data.get('region', '')
        allergies = data.get('allergies', '')
        diet_restrictions = data.get('diet_restrictions', '')
        gym_schedule = data.get('gym_schedule', '')
        hours_of_sleep = int(data['hours_of_sleep']) if data.get('hours_of_sleep', '0').isdigit() else 0
        very_light_movements = int(data['very_light_movements']) if data.get('very_light_movements', '0').isdigit() else 0
        light_movements = int(data['light_movements']) if data.get('light_movements', '0').isdigit() else 0
        moderate_movements = int(data['moderate_movements']) if data.get('moderate_movements', '0').isdigit() else 0
        heavy_movements = int(data['heavy_movements']) if data.get('heavy_movements', '0').isdigit() else 0

        def calculate_activity_factor(hours_of_sleep, very_light_movements, light_movements, moderate_movements, heavy_movements):
            total_hours = hours_of_sleep + very_light_movements + light_movements + moderate_movements + heavy_movements
            if total_hours == 0:
                return 1.2  # Assuming sedentary if no activity is provided
            activity_factor = (hours_of_sleep * 1.2 + very_light_movements * 1.375 + light_movements * 1.55 + moderate_movements * 1.725 + heavy_movements * 1.9) / total_hours
            return activity_factor

        def calculate_calories(sex, weight_kg, height_cm, age, hours_of_sleep, very_light_movements, light_movements, moderate_movements, heavy_movements,workout_goal):
            if sex.lower() == 'male':
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
            elif sex.lower() == 'female':
                bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
            else:
                return "Invalid sex input"
            
            activity_factor = calculate_activity_factor(hours_of_sleep, very_light_movements, light_movements, moderate_movements, heavy_movements)
            maintenance_calories = bmr * activity_factor
    
            # Adjust calories based on goal
            if workout_goal.lower() == 'lose weight':
                goal_calories = maintenance_calories - 500  # Creating a deficit of 500 calories
            elif workout_goal.lower() == 'gain muscle':
                goal_calories = maintenance_calories + 500  # Creating a surplus of 500 calories
            else:
                goal_calories = maintenance_calories  # Maintenance
            
            return goal_calories
        # Example usage
        calories_needed = calculate_calories(sex, weight, height, age, hours_of_sleep, very_light_movements, light_movements, moderate_movements, heavy_movements,workout_goal)  # Adjust the hours spent in each activity level accordingly
        print(f"Daily calorie requirement: {calories_needed} calories")

        def assign_calories(gym_schedule, total_calories):
            total_calories = float(total_calories)
            # Default configuration; adjust as needed
            parts = ["breakfast", "snack1", "lunch", "snack2", "dinner"]
            post_workout_percent = 0.15
            pre_workout_percent = 0.03
            snack1_percent = 0.05
            snack2_percent = 0.05
            breakfast_percent = 0.20
            lunch_percent = 0.25
            dinner_percent = 0.25

            # Configure based on gym_schedule
            if gym_schedule == "morning":
                parts = ["pre-workout", "post-workout", "breakfast", "snack1", "lunch", "snack2", "dinner"]
                # Percentages are already set to default values that match the morning configuration
            elif gym_schedule == "afternoon":
                parts = ["breakfast", "pre-workout", "post-workout", "snack1", "lunch", "snack2", "dinner"]
                # No changes needed to percentages here either
            elif gym_schedule == "evening":
                parts = ["breakfast", "snack1", "lunch", "pre-workout", "post-workout", "dinner"]
                post_workout_percent = 0.20  # Adjusting for evening configuration
                dinner_percent = 0.22  # Adjusting for evening configuration

            # Calculate calories for each part
            calories = {}
            remaining_calories = total_calories
            for part in parts:
                if part == "post-workout":
                    calories[part] = int(total_calories * post_workout_percent)
                elif part == "pre-workout":
                    calories[part] = int(total_calories * pre_workout_percent)
                elif part == "snack1":
                    calories[part] = int(total_calories * snack1_percent)
                elif part == "snack2":
                    calories[part] = int(total_calories * snack2_percent)
                elif part == "breakfast":
                    calories[part] = int(total_calories * breakfast_percent)
                elif part == "lunch":
                    calories[part] = int(total_calories * lunch_percent)
                elif part == "dinner":
                    calories[part] = int(total_calories * dinner_percent)

                remaining_calories -= calories[part]

            # Distribute remaining calories
            if remaining_calories > 0:
                distributed_calories = remaining_calories // len(parts)
                for part in parts:
                    if part not in ["post-workout", "pre-workout"]:
                        calories[part] += distributed_calories
                        remaining_calories -= distributed_calories

            return calories

        
        calories_distribution = assign_calories(gym_schedule, calories_needed)
        for part, cal in calories_distribution.items():
            print(part, ": ", cal, " calories")

        return {'calories_needed': round(calories_needed-1000,2)}

        # Output the part calories dictionary
        #print(part_calories)


def assign_calories(gym_schedule, total_calories):
    total_calories = float(total_calories)
    # Default configuration; adjust as needed
    parts = ["breakfast", "snack1", "lunch", "snack2", "dinner"]
    post_workout_percent = 0.15
    pre_workout_percent = 0.03
    snack1_percent = 0.05
    snack2_percent = 0.05
    breakfast_percent = 0.20
    lunch_percent = 0.25
    dinner_percent = 0.25

    # Configure based on gym_schedule
    if gym_schedule == "morning":
        parts = ["pre-workout", "post-workout", "breakfast", "snack1", "lunch", "snack2", "dinner"]
        # Percentages are already set to default values that match the morning configuration
    elif gym_schedule == "afternoon":
        parts = ["breakfast", "pre-workout", "post-workout", "snack1", "lunch", "snack2", "dinner"]
        # No changes needed to percentages here either
    elif gym_schedule == "evening":
        parts = ["breakfast", "snack1", "lunch", "pre-workout", "post-workout", "dinner"]
        post_workout_percent = 0.20  # Adjusting for evening configuration
        dinner_percent = 0.22  # Adjusting for evening configuration

    # Calculate calories for each part
    calories = {}
    remaining_calories = total_calories
    for part in parts:
        if part == "post-workout":
            calories[part] = int(total_calories * post_workout_percent)
        elif part == "pre-workout":
            calories[part] = int(total_calories * pre_workout_percent)
        elif part == "snack1":
            calories[part] = int(total_calories * snack1_percent)
        elif part == "snack2":
            calories[part] = int(total_calories * snack2_percent)
        elif part == "breakfast":
            calories[part] = int(total_calories * breakfast_percent)
        elif part == "lunch":
            calories[part] = int(total_calories * lunch_percent)
        elif part == "dinner":
            calories[part] = int(total_calories * dinner_percent)

        remaining_calories -= calories[part]

    # Distribute remaining calories
    if remaining_calories > 0:
        distributed_calories = remaining_calories // len(parts)
        for part in parts:
            if part not in ["post-workout", "pre-workout"]:
                calories[part] += distributed_calories
                remaining_calories -= distributed_calories

    return calories
def recommend_meal_with_details(diet_dict, meal_type, nutritional_info_filename):
    meal_calories = diet_dict[meal_type]
    suitable_items = []
    with open(nutritional_info_filename, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            item = row['FoodItem'].lower()
            calories = int(row['Calories'])
            if calories <= meal_calories:
                suitable_items.append((item, calories))
                
    if suitable_items:
        selected_item, item_calories = random.choice(suitable_items)
        meal_calories -= item_calories
        new_diet_dict = diet_dict.copy()
        new_diet_dict[meal_type] = meal_calories
        meal_list = [{'name': selected_item, 'calories': item_calories}]
        return new_diet_dict, meal_list
    else:
        return diet_dict, []

# Utility function to get food details by region
def get_food_details_by_region(region, filename='Server\\f1_lowercase.csv'):
    food_details_list = []
    with open(filename, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Cuisines'].lower() == region.lower():
                food_details = {key: row[key] for key in row}
                food_details_list.append(food_details)
    return food_details_list

def getStructuredResponse(updated_diet_dict, meal_details,food_details_list,region,total_calories):
    generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
    }

    safety_settings = [
    {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]
    
    model = genai.GenerativeModel(model_name="gemini-1.0-pro",
    safety_settings=safety_settings)
    
    prompt_parts = [
    f"""
I have a list of meals with specific calorie requirements and a list of foods with their calorie content. 
I need to distribute these foods into meal categories ( breakfast, snack1, lunch, snack2, dinner) based on their calorie content to meet the daily calorie distribution as closely as possible. before distributing , know which of these foods , most justify the time of day it must be conumed
Also if food are too odd for that session , give some other food , which is almost the same and goes well with {region}
Calorie requirements per meal:
{updated_diet_dict}

List of foods with their calorie content:
{food_details_list}

Organize the food items into the meal categories to meet the calorie requirements. Do not exceed the calorie limit 
Make sure {total_calories} is achieved by overal food items its very very important

"""
    
    ]
   

    response = model.generate_content(prompt_parts)
    generated_text = response.text if response.text else "No content generated."
    print(response.text)
    return {'generatedText' : generated_text}

# Flask route to generate a diet plan
@app.route('/generateDietPlan', methods=['GET', 'POST'])
def generate_diet_plan():
    if request.method == 'POST':
        data = request.get_json()
        gym_schedule = data.get('gym_schedule','')
        region = data.get('region','South Bangalore')
        total_calories = data.get('total_calories','')
        meal_types = ['breakfast', 'snack1', 'lunch', 'dinner']
        nutritional_info_files = {
            'breakfast': 'Server\\orange_veges_nutritional_info.csv',
            'snack1': 'Server\\yellow_green_veges_fruits_nutritional_info.csv',
            'lunch': 'Server\\red_veges_nutritional_info.csv',
            'dinner': 'Server\\green_vegetables_nutritional_info.csv'
        }
        

        updated_diet_dict = assign_calories(gym_schedule,total_calories)
        meal_details = {}

        for meal_type in meal_types:
            updated_diet_dict, meal_list = recommend_meal_with_details(updated_diet_dict, meal_type, nutritional_info_files[meal_type])
            meal_details[meal_type] = meal_list

        # The following region-based food details are for demonstration.
        # In a real scenario, the region could be passed in the request.
        food_details_list = get_food_details_by_region(region)
        structuredResponse = getStructuredResponse(updated_diet_dict, meal_details,food_details_list,region,total_calories)
        return jsonify(structuredResponse)
    else:
        return 'This route supports POST requests only.'


    
@app.route('/')
def server_status():
    return 'Server is up and running'







if __name__ == '__main__':
    app.run(debug=True)