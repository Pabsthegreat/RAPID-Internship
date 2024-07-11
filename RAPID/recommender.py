import pandas as pd
import os
from datetime import datetime, timedelta

class FoodRecommender:
    def __init__(self, food_database_path):
        self.food_database_path = food_database_path
        self.base_increase_per_gram = 4  # Amount blood sugar goes up per gram of carbs
        self.meal_timing_factors = {
            "breakfast": 0.8,
            "lunch": 1.0,
            "dinner": 1.2,
            "snack": 0.6  # Adjusted meal timing factor for snacks
        }
        self.activity_factors = {
            "sedentary": 1.2,
            "moderate": 1.55,
            "frequent": 1.75,
            "super": 1.9
        }
        self.current_preprandial = None
        self.isf = None
        self.hba1c = None
        self.last_meal_time = None
        self.cumulative_macros = {
            "protein": 0,
            "carbs": 0,
            "fat": 0,
            "calories": 0
        }

    def get_meal_suffix(self, meal_time):
        return {
            "breakfast": "_b",
            "lunch": "_l",
            "dinner": "_d",
            "snack": "_s"
        }[meal_time]

    def load_food_data(self, region, subregion, meal_time):
        meal_suffix = self.get_meal_suffix(meal_time)
        region_path = self.food_database_path+"\\"+ region +"\\"+subregion+"\\"+subregion+meal_suffix+".csv"
        
        if os.path.exists(region_path):
            print(f"Loading data from: {region_path}")
            return pd.read_csv(region_path)
        else:
            print(f"File not found: {region_path}")
            return pd.DataFrame()

    def list_subregions(self, region):
        region_path = os.path.join(self.food_database_path, region)
        subregions = [name for name in os.listdir(region_path) if os.path.isdir(os.path.join(region_path, name))]
        print(f"Subregions for {region}: {subregions}")
        return subregions
    
    def filter_foods(self, foods, preference, carb_limit):
        if preference == "vegetarian":
            filtered_foods = foods[foods["Type"] == "veg"]
        elif preference == "non-veg":
            filtered_foods = foods[foods["Type"].isin(["non-veg", "egg"])]
        elif preference == "eggitarian":
            filtered_foods = foods[foods["Type"].isin(["veg", "egg"])]
        else:
            filtered_foods = foods
        
        # Further filter based on carbohydrate limit
        filtered_foods = filtered_foods[filtered_foods["Carbs (in g)"] <= carb_limit]
        
        return filtered_foods

    def calculate_postprandial(self, preprandial_level, carbs_consumed, activity_level, meal_time):
        activity_factor = self.activity_factors[activity_level]
        meal_timing_factor = self.meal_timing_factors[meal_time]
        
        if self.isf is None or self.hba1c is None:
            raise ValueError("ISF and HbA1c must be set before calculating postprandial levels.")
        
        hba1c_factor = 1 + ((self.hba1c - 5) / 10)  # Adjusted based on HbA1c level
        
        return preprandial_level + (carbs_consumed * self.base_increase_per_gram * activity_factor * meal_timing_factor * hba1c_factor) / self.isf

    def calculate_preprandial(self, postprandial_level, time_between_meals):
        return postprandial_level - (time_between_meals * self.isf)

    def recommend_meal(self, preprandial_level, preference, region, subregion, meal_time, activity_level, macro_limits):
        foods = self.load_food_data(region, subregion, meal_time)
        if foods.empty:
            return None, "No suitable foods found in this region and subregion. Please choose another subregion."

        carb_limit = (macro_limits["postprandial_target"] - preprandial_level) / (
                self.base_increase_per_gram * self.activity_factors[activity_level] * self.meal_timing_factors[meal_time]
        )

        filtered_foods = self.filter_foods(foods, preference, carb_limit)
        if filtered_foods.empty:
            print(f"No foods found under carb limit {carb_limit}.")
            return None, "No suitable foods found in this subregion under carbohydrate limits. Please choose another subregion."

        for _, food in filtered_foods.iterrows():
            if (self.cumulative_macros["protein"] + food["Proteins (in g)"] <= macro_limits["protein"] and
                self.cumulative_macros["carbs"] + food["Carbs (in g)"] <= macro_limits["carbs"] and
                self.cumulative_macros["fat"] + food["Fats (in g)"] <= macro_limits["fat"] and
                self.cumulative_macros["calories"] + food["Calories (in Cal)"] <= macro_limits["calories"]):
                
                postprandial_level = self.calculate_postprandial(
                    preprandial_level, food["Carbs (in g)"],
                    activity_level, meal_time
                )
                
                self.cumulative_macros["protein"] += food["Proteins (in g)"]
                self.cumulative_macros["carbs"] += food["Carbs (in g)"]
                self.cumulative_macros["fat"] += food["Fats (in g)"]
                self.cumulative_macros["calories"] += food["Calories (in Cal)"]
                
                # Print additional nutrients
                print(f"Recommended {meal_time.capitalize()}: {food['Food item']}")
                print(f"Protein: {food['Proteins (in g)']} g, Carbs: {food['Carbs (in g)']} g, Fat: {food['Fats (in g)']} g")
                print(f"Calories: {food['Calories (in Cal)']}, Calcium: {food['Calcium (in mg)']} mg, Fiber: {food['Fiber (in g)']} g\n")
                
                return food.to_dict(), postprandial_level
            else:
                print(f"Skipping food: {food['Food item']}")
                print(f"Reason: Protein limit exceeded: {self.cumulative_macros['protein'] + food['Proteins (in g)']} > {macro_limits['protein']}")

        return None, "No suitable foods found within macro limits. Please choose another subregion or regenerate the meal."

    def set_isf(self, isf):
        self.isf = isf
    
    def set_hba1c(self, hba1c):
        self.hba1c = hba1c
    
    def set_last_meal_time(self, last_meal_time):
        self.last_meal_time = last_meal_time

    def calculate_time_between_meals(self, current_meal_time):
        if self.last_meal_time is None:
            raise ValueError("Last meal time must be set to calculate time between meals.")
        
        current_time = datetime.strptime(current_meal_time, "%I%p")
        last_time = datetime.strptime(self.last_meal_time, "%I%p")
        
        # Calculate time difference in hours
        time_difference = (current_time - last_time).seconds / 3600
        
        return time_difference

    def recommend_daily_meals(self, initial_preprandial, preference, regions, meal_times, activity_level, macro_limits):
        self.current_preprandial = initial_preprandial
        daily_meals = {}
        
        for meal_time in meal_times:
            while True:
                print(f"Choose a region: {regions}")
                region = input()
                subregions = self.list_subregions(region)
                print(f"Choose a subregion for {region} for {meal_time}: {subregions}")
                subregion = input()
                
                recommended_food, postprandial_level = self.recommend_meal(
                    self.current_preprandial, preference, region, subregion, meal_time, activity_level, macro_limits
                )
                
                if recommended_food:
                    daily_meals[meal_time] = recommended_food
                    self.current_preprandial = self.calculate_preprandial(
                        postprandial_level, self.calculate_time_between_meals(meal_time)
                    )
                    self.set_last_meal_time(meal_time)
                    print(f"Recommended {meal_time.capitalize()}: {recommended_food['Food item']}")
                    print(f"Protein: {recommended_food['Proteins (in g)']} g, Carbs: {recommended_food['Carbs (in g)']} g, Fat: {recommended_food['Fats (in g)']} g")
                    print(f"Calories: {recommended_food['Calories (in Cal)']}, Calcium: {recommended_food['Calcium (in mg)']} mg, Fiber: {recommended_food['Fiber (in g)']} g\n")
                    break
                else:
                    print("Please choose another subregion or regenerate the meal.")
        
        return daily_meals

# Example Usage
food_database_path = "RAPID\\datasets\\food"
recommender = FoodRecommender(food_database_path)

# Set ISF and HbA1c (example values)
recommender.set_isf(50)  # Example ISF value
recommender.set_hba1c(6.5)  # Example HbA1c value

initial_preprandial = 100  # Example initial preprandial level
preference = "vegetarian"  # Dietary preference
regions = ["Karnataka", "Andhra", "Kerala", "Maharashtra", "Tamil Nadu", "general"]  # Available regions
meal_times = ["breakfast", "lunch", "dinner", "snack"]  # Example meal times
activity_level = "moderate"  # Activity level
macro_limits = {
    "protein": 50,
    "carbs": 60,
    "fat": 30,
    "calories": 800,
    "postprandial_target": 500 # Example target for postprandial blood sugar level
}

recommended_meals = recommender.recommend_daily_meals(initial_preprandial, preference, regions, meal_times, activity_level, macro_limits)

















# import pandas as pd
# from datetime import datetime, timedelta

# class FoodRecommender:
#     def __init__(self, food_database_path):
#         self.food_database_path = food_database_path
#         self.base_increase_per_gram = 4  # Amount blood sugar goes up per gram of carbs
#         self.meal_timing_factors = {
#             "breakfast": 0.8,
#             "lunch": 1.0,
#             "dinner": 1.2,
#             "snack": 0.6  # Adjusted meal timing factor for snacks
#         }
#         self.activity_factors = {
#             "sedentary": 1.2,
#             "moderate": 1.55,
#             "frequent": 1.75,
#             "super": 1.9
#         }
#         self.current_preprandial = None
#         self.isf = None
#         self.hba1c = None
#         self.last_meal_time = None
#         self.cumulative_macros = {
#             "protein": 0,
#             "carbs": 0,
#             "fat": 0,
#             "calories": 0
#         }
#         self.meal_times = {
#             "breakfast": None,
#             "lunch": None,
#             "dinner": None,
#             "snack": None
#         }

#     def load_food_data(self, region, subregion, meal_time_suffix):
#         file_path = f"{self.food_database_path}/{region}/{subregion}_{meal_time_suffix}.csv"
#         try:
#             return pd.read_csv(file_path)
#         except FileNotFoundError as e:
#             raise FileNotFoundError(f"File not found: {file_path}")
#         except pd.errors.ParserError as e:
#             raise ValueError(f"Error parsing data from {file_path}. Check the file format.")

#     def filter_foods(self, foods, preference, carb_limit):
#         if preference == "vegetarian":
#             filtered_foods = foods[foods["Type"] == "veg"]
#         elif preference == "non-veg":
#             filtered_foods = foods[foods["Type"].isin(["non-veg", "egg"])]
#         elif preference == "eggitarian":
#             filtered_foods = foods[foods["Type"].isin(["veg", "egg"])]

#         # Further filter based on carbohydrate limit
#         filtered_foods = filtered_foods[filtered_foods["Carbs (in g)"] <= carb_limit]

#         return filtered_foods

#     def calculate_postprandial(self, preprandial_level, carbs_consumed, activity_level, meal_time):
#         activity_factor = self.activity_factors[activity_level]
#         meal_timing_factor = self.meal_timing_factors[meal_time]

#         if self.isf is None or self.hba1c is None:
#             raise ValueError("ISF and HbA1c must be set before calculating postprandial levels.")

#         hba1c_factor = 1 + ((self.hba1c - 5) / 10)  # Adjusted based on HbA1c level

#         return preprandial_level + (carbs_consumed * self.base_increase_per_gram * activity_factor * meal_timing_factor * hba1c_factor) / self.isf

#     def calculate_preprandial(self, postprandial_level, carbs_consumed, activity_level, meal_time):
#         activity_factor = self.activity_factors[activity_level]
#         meal_timing_factor = self.meal_timing_factors[meal_time]

#         if self.isf is None or self.hba1c is None:
#             raise ValueError("ISF and HbA1c must be set before calculating preprandial levels.")

#         hba1c_factor = 1 + ((self.hba1c - 5) / 10)  # Adjusted based on HbA1c level

#         return (postprandial_level - (carbs_consumed * self.base_increase_per_gram * activity_factor * meal_timing_factor * hba1c_factor)) * self.isf

#     def set_meal_times(self):
#         for meal in self.meal_times:
#             while True:
#                 try:
#                     time_str = input(f"Enter time for {meal} (e.g., 1PM): ").strip().upper()
#                     meal_time = datetime.strptime(time_str, "%I%p").strftime("%I%p")
#                     self.meal_times[meal] = meal_time
#                     break
#                 except ValueError:
#                     print("Invalid time format. Please enter time in 12-hour format like '1PM'.")

#     def recommend_meal(self, preprandial_level, preference, region, subregion, meal_time_suffix, activity_level, macro_limits):
#         try:
#             foods = self.load_food_data(region, subregion, meal_time_suffix)
#         except FileNotFoundError as e:
#             return None, str(e)
#         except ValueError as e:
#             return None, str(e)

#         if foods.empty:
#             return None, f"No suitable foods found in {subregion} for {meal_time_suffix}. Please choose another subregion."

#         carb_limit = (macro_limits["postprandial_target"] - preprandial_level) / (
#                 self.base_increase_per_gram * self.activity_factors[activity_level] * self.meal_timing_factors[meal_time_suffix[:-2]]
#         )

#         filtered_foods = self.filter_foods(foods, preference, carb_limit)

#         if filtered_foods.empty:
#             return None, "No suitable foods found within macro limits. Please choose another subregion or regenerate the meal."

#         for _, food in filtered_foods.iterrows():
#             if (self.cumulative_macros["protein"] + food["Proteins (in g)"] <= macro_limits["protein"] and
#                 self.cumulative_macros["carbs"] + food["Carbs (in g)"] <= macro_limits["carbs"] and
#                 self.cumulative_macros["fat"] + food["Fats (in g)"] <= macro_limits["fat"] and
#                 self.cumulative_macros["calories"] + food["Calories (in Cal)"] <= macro_limits["calories"]):
                
#                 postprandial_level = self.calculate_postprandial(
#                     preprandial_level, food["Carbs (in g)"], activity_level, meal_time_suffix[:-2]
#                 )
                
#                 self.cumulative_macros["protein"] += food["Proteins (in g)"]
#                 self.cumulative_macros["carbs"] += food["Carbs (in g)"]
#                 self.cumulative_macros["fat"] += food["Fats (in g)"]
#                 self.cumulative_macros["calories"] += food["Calories (in Cal)"]
                
#                 # Print additional nutrients
#                 print(f"Recommended {meal_time_suffix[:-2].capitalize()}: {food['Food item']}")
#                 print(f"Protein: {food['Proteins (in g)']} g, Carbs: {food['Carbs (in g)']} g, Fat: {food['Fats (in g)']} g")
#                 print(f"Calories: {food['Calories (in Cal)']}, Calcium: {food['Calcium (in mg)']} mg, Fiber: {food['Fiber (in g)']} g\n")
                
#                 return food.to_dict(), postprandial_level
        
#         return None, "No suitable foods found within macro limits. Please choose another subregion or regenerate the meal."

# # Example usage
# if __name__ == "__main__":
#     food_database_path = "RAPID/datasets/food"
#     recommender = FoodRecommender(food_database_path)
    
#     print("Welcome to the Food Recommender System!")
    
#     try:
#         # Set meal times
#         recommender.set_meal_times()
        
#         # Other inputs (initial_preprandial, preference, region, subregion, activity_level, macro_limits)
#         # Example:
#         initial_preprandial = 100
#         preference = "vegetarian"
#         region = "Karnataka"
#         subregion = "southkarnataka"
#         activity_level = "moderate"
#         macro_limits = {
#             "protein": 500,
#             "carbs": 600,
#             "fat": 300,
#             "calories": 8000
#         }
        
#         # Recommend meal
#         for meal, meal_time in recommender.meal_times.items():
#             meal_time_suffix = meal_time + "_b"  # Assuming breakfast for simplicity, can be generalized
#             recommended_meal, postprandial_level = recommender.recommend_meal(
#                 initial_preprandial, preference, region, subregion, meal_time_suffix, activity_level, macro_limits
#             )
#             if recommended_meal:
#                 print(f"Recommended {meal.capitalize()}:", recommended_meal['Food item'])
#                 print(f"Postprandial level: {postprandial_level:.2f}\n")
#             else:
#                 print(f"No recommendation found for {meal.capitalize()}: {postprandial_level}\n")
            
#     except ValueError as ve:
#         print(f"ValueError occurred: {ve}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
