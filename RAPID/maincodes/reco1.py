import pandas as pd
import os
from datetime import datetime
import random
import total_calorie_predictor
class FoodRecommender:
    def __init__(self, food_database_path):
        self.food_database_path = food_database_path
        self.base_increase_per_gram = 3  # Amount blood sugar goes up per gram of carbs
        self.meal_timing_factors = {
            "breakfast": 0.5,
            "lunch": 1.0,
            "dinner": 1.2,
            "snack": 2  # Adjusted meal timing factor for snacks
        }
        self.activity_factors = {
            "sedentary": 1.2,
            "moderate": 1.55,
            "frequent": 1.75,
            "super": 1.9
        }
        self.current_preprandial = None
        self.isf = 30
        self.hba1c = 9
        self.last_meal_time = "6AM"
        self.cumulative_macros = {
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "calories": 0
        }
        self.meal_times = {
            "breakfast": "10AM",
            "lunch": "1PM",
            "snack": "4PM",
            "dinner": "8PM"
        }
        self.recommended = {"breakfast":[],"lunch":[],"dinner":[],"snack":[]}#can keep only 1 at a time
        self.choices = []

    def load_food_data(self, region, subregion, meal_time_suffix):
        file_path = os.path.join(self.food_database_path, region, subregion, f"{subregion}_{meal_time_suffix[0]}.csv")
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except pd.errors.ParserError:
            raise ValueError(f"Error parsing data from {file_path}. Check the file format.")

    def filter_foods(self, foods, preference, carb_limit):
        if preference == "veg":
            filtered_foods = foods[foods["Type"] == "veg"]
        elif preference == "non-veg":
            filtered_foods = foods[foods["Type"].isin(["veg", "non-veg", "egg"])]
        elif preference == "egg":
            filtered_foods = foods[foods["Type"].isin(["veg", "egg"])]

        filtered_foods = filtered_foods[filtered_foods["Carbs (in g)"] <= carb_limit]

        return filtered_foods

    def calculate_postprandial(self, preprandial_level, carbs_consumed, activity_level, meal_time):
        activity_factor = self.activity_factors[activity_level]
        meal_timing_factor = self.meal_timing_factors[meal_time]

        hba1c_factor = 1 + ((self.hba1c - 5) / 10)  # Adjusted based on HbA1c level

        return preprandial_level + (carbs_consumed * self.base_increase_per_gram * activity_factor * meal_timing_factor * hba1c_factor)/(self.isf)

    def calculate_preprandial(self, postprandial_level, time_between_meals):
        if self.isf is None:
            raise ValueError("ISF must be set before calculating preprandial levels.")
        return postprandial_level - (time_between_meals * self.isf)

    def recommend_meal(self, preprandial_level, preference, region, subregion, meal_time_suffix, activity_level, macro_limits):
        try:
            foods = self.load_food_data(region, subregion, meal_time_suffix)
        except FileNotFoundError as e:
            return None
        except ValueError as e:
            return None

        if foods.empty:
            return None #f"No suitable foods found in {subregion} for {meal_time_suffix}. Please choose another subregion."

        carb_limit = (macro_limits["postprandial_target"] - preprandial_level) / (
                self.base_increase_per_gram * self.activity_factors[activity_level] * self.meal_timing_factors[meal_time_suffix]
        )

        print(f"Carb limit for {meal_time_suffix}: {carb_limit} grams")

        filtered_foods = self.filter_foods(foods, preference, carb_limit)

        if filtered_foods.empty:
            return None#, "No suitable foods found within carb limit."

        for _, food in filtered_foods.iterrows():
            reasons = []
            if self.cumulative_macros["protein"] + food["Proteins (in g)"] > macro_limits["protein"]:
                reasons.append(f"Protein limit exceeded: {self.cumulative_macros['protein']} + {food['Proteins (in g)']} > {macro_limits['protein']}")
            if self.cumulative_macros["carbs"] + food["Carbs (in g)"] > macro_limits["carbs"]:
                reasons.append(f"Carb limit exceeded: {self.cumulative_macros['carbs']} + {food['Carbs (in g)']} > {macro_limits['carbs']}")
            if self.cumulative_macros["fat"] + food["Fats (in g)"] > macro_limits["fat"]:
                reasons.append(f"Fat limit exceeded: {self.cumulative_macros['fat']} + {food['Fats (in g)']} > {macro_limits['fat']}")
            if self.cumulative_macros["calories"] + food["Calories (in Cal)"] > macro_limits["calories"]:
                reasons.append(f"Calorie limit exceeded: {self.cumulative_macros['calories']} + {food['Calories (in Cal)']} > {macro_limits['calories']}")
            if reasons:
                continue
            self.choices.append(food)
        recommended_food = self.food_choice_randomiser(meal_time_suffix)
        if recommended_food:
            return self.food_choice_randomiser(meal_time_suffix)
        return None

    def food_choice_randomiser(self,meal_time_suffix):
        random_index = random.randint(0,len(self.choices)-1)
        food = self.choices[random_index]
        while food.to_dict()["Food item"] in self.recommended[meal_time_suffix]:
            random_index = random.randint(0,len(self.choices)-1)
            food = self.choices[random_index]
        return food.to_dict()
    
    def print_food(self,food,meal_time_suffix):
            print(f"Recommended {meal_time_suffix.capitalize()}: {food['Food item']}")
            print(f"Quantity: {food['Quantity']}, Protein: {food['Proteins (in g)']*2} g, Carbs: {food['Carbs (in g)']*2} g, Fat: {food['Fats (in g)']*2} g")
            print(f"Calories: {food['Calories (in Cal)']*2}, Calcium: {food['Calcium (in mg)']*2} mg, Fiber: {food['Fiber (in g)']*2} g\n")
    
    def macro_update(self,food,preprandial_level,meal_time_suffix,activity_level):
        postprandial_level = self.calculate_postprandial(
            preprandial_level, food["Carbs (in g)"], activity_level, meal_time_suffix
        )
        
        self.cumulative_macros["protein"] += food["Proteins (in g)"]
        self.cumulative_macros["carbs"] += food["Carbs (in g)"]
        self.cumulative_macros["fat"] += food["Fats (in g)"]
        self.cumulative_macros["calories"] += food["Calories (in Cal)"]

        return postprandial_level

    def recommend_daily_meals(self, initial_preprandial, preference, activity_level, macro_limits):
        self.current_preprandial = initial_preprandial
        daily_meals = {}

        regions = ["karnataka", "andhra", "kerala", "maharashtra", "tamilnadu", "general"]

        for meal_time in self.meal_times:
            self.choices = []
            while True:
                while True:
                    print(f"\nPlease choose a region for {meal_time.capitalize()}:")
                    for i, region in enumerate(regions):
                        print(f"{i + 1}. {region.capitalize()}")
                    region_choice = int(input("Enter your choice: ")) - 1
                    region = regions[region_choice]

                    subregions = self.get_subregions(region)
                    print(f"\nPlease choose a subregion in {region.capitalize()} for {meal_time.capitalize()}:")
                    for i, subregion in enumerate(subregions):
                        print(f"{i + 1}. {subregion.capitalize()}")
                    subregion_choice = int(input("Enter your choice: ")) - 1
                    subregion = subregions[subregion_choice]

                    recommended_food = self.recommend_meal(
                        self.current_preprandial, preference, region, subregion, meal_time, activity_level, macro_limits
                    )
                    if recommended_food:
                        self.print_food(recommended_food,meal_time)
                    else:
                        print(f"No suitable foods found for {meal_time} in {subregion}")

                    q = input("Do you want to regenrate the meal?:")
                    if q == "n":
                        postprandial_level = self.macro_update(recommended_food,self.current_preprandial,meal_time,activity_level)
                        break
                    elif recommended_food:
                        self.recommended[meal_time] += [recommended_food["Food item"]]
                        continue

                if recommended_food:
                    daily_meals[meal_time] = recommended_food
                    self.current_preprandial = self.calculate_preprandial(
                        postprandial_level, self.calculate_time_between_meals(self.meal_times[meal_time])
                    )
                    self.last_meal_time = self.meal_times[meal_time]
                    break

        return daily_meals,self.cumulative_macros

    def get_subregions(self, region):
        region_path = os.path.join(self.food_database_path, region)
        subregions = [subregion.split("_")[0] for subregion in os.listdir(region_path) if os.path.isdir(os.path.join(region_path, subregion))]
        return list(set(subregions))

    def calculate_time_between_meals(self, next_meal_time):
        time_format = "%I%p"
        last_meal_time = datetime.strptime(self.last_meal_time, time_format)
        next_meal_time = datetime.strptime(next_meal_time, time_format)
        return (next_meal_time - last_meal_time).seconds / 3600  # Convert to hours

# Example usage
food_database_path = "RAPID\\datasets\\food"
recommender = FoodRecommender(food_database_path)
initial_preprandial = 90
preference = "non-veg"
activity_level = "frequent"
macro_limits = {
    "protein": 100,  # in grams
    "carbs": 200,  # in grams
    "fat": 70,  # in grams
    "calories": 2000,  # in calories
    "postprandial_target": 180  # Target postprandial level
}

# Call the method
daily_meals, day_macros= recommender.recommend_daily_meals(initial_preprandial, preference, activity_level, macro_limits)
for i in daily_meals:
    print(i+":")
    for j in daily_meals[i]:
        print(str(j)+": "+str(daily_meals[i][j]))
    print("\n")

for i in day_macros:
    print(i+": "+str(day_macros[i]*2))