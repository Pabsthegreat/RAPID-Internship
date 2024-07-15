import pandas as pd
import os
from datetime import datetime

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
        self.isf = 60
        self.hba1c = 9
        self.last_meal_time = "10PM"
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
    #opens a file 
    def load_food_data(self, region, subregion, meal_time_suffix):
        file_path = os.path.join(self.food_database_path, region, subregion, f"{subregion}_{meal_time_suffix[0]}.csv")
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except pd.errors.ParserError:
            raise ValueError(f"Error parsing data from {file_path}. Check the file format.")

    def filter_foods(self, foods, preference, carb_limit):
        if preference == "vegetarian":
            filtered_foods = foods[foods["Type"] == "veg"]
        elif preference == "non-veg":
            filtered_foods = foods[foods["Type"].isin(["veg", "non-veg", "egg"])]
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
        if self.isf is None:
            raise ValueError("ISF must be set before calculating preprandial levels.")

        return postprandial_level - (time_between_meals * self.isf)

    def set_meal_times(self):
        for meal in self.meal_times:
            while True:
                try:
                    time_str = input(f"Enter time for {meal} (e.g., 1PM): ").strip().upper()
                    meal_time = datetime.strptime(time_str, "%I%p").strftime("%I%p")
                    self.meal_times[meal] = meal_time
                    break
                except ValueError:
                    print("Invalid time format. Please enter time in 12-hour format like '1PM'.")

    def recommend_meal(self, preprandial_level, preference, region, subregion, meal_time_suffix, activity_level, macro_limits):
        try:
            foods = self.load_food_data(region, subregion, meal_time_suffix)
        except FileNotFoundError as e:
            return None, str(e)
        except ValueError as e:
            return None, str(e)

        if foods.empty:
            return None, f"No suitable foods found in {subregion} for {meal_time_suffix}. Please choose another subregion."

        carb_limit = (macro_limits["postprandial_target"] - preprandial_level) / (
                self.base_increase_per_gram * self.activity_factors[activity_level] * self.meal_timing_factors[meal_time_suffix]
        )

        filtered_foods = self.filter_foods(foods, preference, carb_limit)

        if filtered_foods.empty:
            return None, "No suitable foods found within macro limits in recommend meal. Please choose another subregion or regenerate the meal."

        for _, food in filtered_foods.iterrows():
            if (self.cumulative_macros["protein"] + food["Proteins (in g)"] <= macro_limits["protein"] and
                self.cumulative_macros["carbs"] + food["Carbs (in g)"] <= macro_limits["carbs"] and
                self.cumulative_macros["fat"] + food["Fats (in g)"] <= macro_limits["fat"] and
                self.cumulative_macros["calories"] + food["Calories (in Cal)"] <= macro_limits["calories"]):
                
                postprandial_level = self.calculate_postprandial(
                    preprandial_level, food["Carbs (in g)"], activity_level, meal_time_suffix
                )

                self.cumulative_macros["protein"] += food["Proteins (in g)"]
                self.cumulative_macros["carbs"] += food["Carbs (in g)"]
                self.cumulative_macros["fat"] += food["Fats (in g)"]
                self.cumulative_macros["calories"] += food["Calories (in Cal)"]

                return food.to_dict(), postprandial_level

        return None, "No suitable foods found within macro limits after filtering. Please choose another subregion or regenerate the meal."

    # def recommend_daily_meals(self, initial_preprandial, preference, regions, meal_times, activity_level, macro_limits):
    #     self.current_preprandial = initial_preprandial
    #     daily_meals = {}
    #     for meal_time in meal_times:
    #         meal_recommended = False
    #         for region in regions:
    #             subregions = self.get_subregions(region)
    #             for subregion in subregions:
    #                 recommended_food, postprandial_level = self.recommend_meal(
    #                     self.current_preprandial, preference, region, subregion, meal_time, activity_level, macro_limits
    #                 )

    #                 if recommended_food:
    #                     daily_meals[meal_time] = recommended_food
    #                     self.current_preprandial = self.calculate_preprandial(
    #                         postprandial_level, self.calculate_time_between_meals(self.meal_times[meal_time])
    #                     )
    #                     self.last_meal_time = self.meal_times[meal_time]
    #                     meal_recommended = True
    #                     break

    #             if meal_recommended:
    #                 break

    #         if not meal_recommended:
    #             print(f"No suitable foods found for {meal_time}. Please adjust your preferences or limits.")
    #             break

    #     return daily_meals
    def recommend_daily_meals(self, initial_preprandial, preference, meal_times, activity_level, macro_limits):
        self.current_preprandial = initial_preprandial
        daily_meals = {}
        
        regions = os.listdir(self.food_database_path)
        print("Available regions:")
        for idx, region in enumerate(regions):
            print(f"{idx + 1}. {region.capitalize()}")

        for meal_time in meal_times:
            while True:
                try:
                    region_idx = int(input(f"Select region for {meal_time} (enter number): ").strip()) - 1
                    if 0 <= region_idx < len(regions):
                        region = regions[region_idx]
                        break
                    else:
                        print("Invalid selection. Please choose a valid region number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            subregions = self.get_subregions(region)
            print(f"Available subregions in {region.capitalize()}:")
            for idx, subregion in enumerate(subregions):
                print(f"{idx + 1}. {subregion.capitalize()}")

            while True:
                try:
                    subregion_idx = int(input(f"Select subregion for {meal_time} (enter number): ").strip()) - 1
                    if 0 <= subregion_idx < len(subregions):
                        subregion = subregions[subregion_idx]
                        break
                    else:
                        print("Invalid selection. Please choose a valid subregion number.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            
            recommended_food, postprandial_level = self.recommend_meal(
                self.current_preprandial, preference, region, subregion, meal_time, activity_level, macro_limits
            )
            
            if recommended_food:
                daily_meals[meal_time] = recommended_food
                self.current_preprandial = self.calculate_preprandial(
                    postprandial_level, self.calculate_time_between_meals(self.meal_times[meal_time])
                )
                self.last_meal_time = self.meal_times[meal_time]
                print(f"Recommended {meal_time.capitalize()}: {recommended_food['Food item']}")
                print(f"Protein: {recommended_food['Proteins (in g)']} g, Carbs: {recommended_food['Carbs (in g)']} g, Fat: {recommended_food['Fats (in g)']} g")
                print(f"Calories: {recommended_food['Calories (in Cal)']}, Calcium: {recommended_food['Calcium (in mg)']} mg, Fiber: {recommended_food['Fiber (in g)']} g\n")
            else:
                print(f"No suitable foods found for {meal_time}. {postprandial_level}")

        return daily_meals

    def get_subregions(self, region):
        region_path = os.path.join(self.food_database_path, region)
        subregions = [subregion.split("_")[0] for subregion in os.listdir(region_path) if os.path.isdir(os.path.join(region_path, subregion))]
        return list(set(subregions))

    def set_isf(self, isf):
        self.isf = isf
    
    def set_hba1c(self, hba1c):
        self.hba1c = hba1c

    def calculate_time_between_meals(self, current_meal_time):
        current_time = datetime.strptime(current_meal_time, "%I%p")
        last_time = datetime.strptime(self.last_meal_time, "%I%p")
        
        # Calculate time difference in hours
        time_difference = (current_time - last_time).seconds / 3600
        
        return time_difference

# Example usage
if __name__ == "__main__":
    food_database_path = "RAPID\\datasets\\food"
    recommender = FoodRecommender(food_database_path)
    
    print("Welcome to the Food Recommender System!")
    
    try:
        recommender.set_isf(60)  # float(input("Enter Insulin Sensitivity Factor (ISF): "))
        recommender.set_hba1c(9)  # float(input("Enter HbA1c: "))
    except ValueError as e:
        print(str(e))
        exit(1)
    
    initial_preprandial = 100  # float(input("Enter initial preprandial blood sugar level: "))
    preference = "vegetarian"  # input("Enter dietary preference (vegetarian, non-veg, eggitarian): ").lower()
    activity_level = "moderate"  # input("Enter activity level (sedentary, moderate, frequent, super): ").lower()
    
    macro_limits = {
        "protein": 100,  # float(input("Enter maximum protein intake (in g): ")),
        "carbs": 250,  # float(input("Enter maximum carb intake (in g): ")),
        "fat": 50,  # float(input("Enter maximum fat intake (in g): ")),
        "calories": 1950,  # float(input("Enter maximum calorie intake: ")),
        "postprandial_target": 180  # float(input("Enter postprandial blood sugar target level: "))
    }
    
    regions = ["karnataka", "andhra", "kerala", "maharashtra", "tamilnadu", "general"]
    meal_times = ["breakfast", "lunch", "dinner", "snack"]
    
    daily_meals = recommender.recommend_daily_meals(initial_preprandial, preference, meal_times, activity_level, macro_limits)
    
    print("\nRecommended Daily Meals:")
    for meal_time, meal_details in daily_meals.items():
        print(f"{meal_time.capitalize()}: {meal_details['Food item']}")
        print(f"Protein: {meal_details['Proteins (in g)']} g, Carbs: {meal_details['Carbs (in g)']} g, Fat: {meal_details['Fats (in g)']} g")
        print(f"Calories: {meal_details['Calories (in Cal)']}, Calcium: {meal_details['Calcium (in mg)']} mg, Fiber: {meal_details['Fiber (in g)']} g\n")


# def recommend_daily_meals(self, initial_preprandial, preference, meal_times, activity_level, macro_limits):
#     self.current_preprandial = initial_preprandial
#     daily_meals = {}
    
#     regions = os.listdir(self.food_database_path)
#     print("Available regions:")
#     for idx, region in enumerate(regions):
#         print(f"{idx + 1}. {region.capitalize()}")

#     for meal_time in meal_times:
#         while True:
#             try:
#                 region_idx = int(input(f"Select region for {meal_time} (enter number): ").strip()) - 1
#                 if 0 <= region_idx < len(regions):
#                     region = regions[region_idx]
#                     break
#                 else:
#                     print("Invalid selection. Please choose a valid region number.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")
        
#         subregions = self.get_subregions(region)
#         print(f"Available subregions in {region.capitalize()}:")
#         for idx, subregion in enumerate(subregions):
#             print(f"{idx + 1}. {subregion.capitalize()}")

#         while True:
#             try:
#                 subregion_idx = int(input(f"Select subregion for {meal_time} (enter number): ").strip()) - 1
#                 if 0 <= subregion_idx < len(subregions):
#                     subregion = subregions[subregion_idx]
#                     break
#                 else:
#                     print("Invalid selection. Please choose a valid subregion number.")
#             except ValueError:
#                 print("Invalid input. Please enter a number.")
        
#         recommended_food, postprandial_level = self.recommend_meal(
#             self.current_preprandial, preference, region, subregion, meal_time, activity_level, macro_limits
#         )
        
#         if recommended_food:
#             daily_meals[meal_time] = recommended_food
#             self.current_preprandial = self.calculate_preprandial(
#                 postprandial_level, self.calculate_time_between_meals(self.meal_times[meal_time])
#             )
#             self.last_meal_time = self.meal_times[meal_time]
#             print(f"Recommended {meal_time.capitalize()}: {recommended_food['Food item']}")
#             print(f"Protein: {recommended_food['Proteins (in g)']} g, Carbs: {recommended_food['Carbs (in g)']} g, Fat: {recommended_food['Fats (in g)']} g")
#             print(f"Calories: {recommended_food['Calories (in Cal)']}, Calcium: {recommended_food['Calcium (in mg)']} mg, Fiber: {recommended_food['Fiber (in g)']} g\n")
#         else:
#             print(f"No suitable foods found for {meal_time}. {postprandial_level}")

#     return daily_meals

