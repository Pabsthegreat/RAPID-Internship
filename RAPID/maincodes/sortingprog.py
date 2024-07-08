import pandas as pd
from itertools import product

# Load the food database
food_db = pd.read_csv('Andhra.csv')

# Filter for main dishes that need side dishes, side dishes, and independent dishes
main_dishes = food_db[food_db['Needside'] == True]
side_dishes = food_db[food_db['Side'] == True]
independent_dishes = food_db[(food_db['Needside'] == False) & (food_db['Side'] == False)]

# Generate combinations of main dishes and side dishes
combos = list(product(main_dishes['Food item'], side_dishes['Food item']))

# Load the progress index if it exists
try:
    with open('progress.txt', 'r') as file:
        start_index = int(file.read().strip())
except FileNotFoundError:
    start_index = 0

# Function to manually rank food combinations
def manual_rank_combinations(combos, db, start_index):
    ranked_combos = []
    for i, combo in enumerate(combos[start_index:], start=start_index):
        food1 = db[db['Food item'] == combo[0]].iloc[0]
        food2 = db[db['Food item'] == combo[1]].iloc[0]
        
        meal_timings1 = food1['MealTimings'].split(',')
        meal_timings2 = food2['MealTimings'].split(',')
        common_timings = set(meal_timings1) & set(meal_timings2)
        
        for timing in common_timings:
            print(f"\nCombination {i + 1}: {food1['Food item']} + {food2['Food item']}")
            print(f"Cuisines: {food1['Cuisines']}, Type: {food1['Type']}, MealTimings: {timing}")
            print(f"Food 1 - Carbs: {food1['Carbs (in g)']}g, Proteins: {food1['Protiens (in g)']}g, Fats: {food1['Fats (in g)']}g, GI: {food1['GI']}")
            print(f"Food 2 - Carbs: {food2['Carbs (in g)']}g, Proteins: {food2['Protiens (in g)']}g, Fats: {food2['Fats (in g)']}g, GI: {food2['GI']}")
            
            rank_score = float(input("Enter the rank score for this combination (0-5): "))
            
            if rank_score > 0:
                combined_type = 'non veg' if food1['Type'] == 'non veg' or food2['Type'] == 'non veg' else 'veg'
                meat_type = 0 if combined_type == 'veg' else food1['meat_type'] if food1['Type'] == 'non veg' else food2['meat_type']
                
                ranked_combos.append({
                    'Cuisines': food1['Cuisines'],
                    'Food item': f"{food1['Food item']} + {food2['Food item']}",
                    'Type': combined_type,
                    'MealTimings': timing,
                    'Quantity': f"{food1['Quantity']} + {food2['Quantity']}",
                    'Protiens (in g)': food1['Protiens (in g)'] + food2['Protiens (in g)'],
                    'Carbs (in g)': food1['Carbs (in g)'] + food2['Carbs (in g)'],
                    'Fats (in g)': food1['Fats (in g)'] + food2['Fats (in g)'],
                    'Fiber (in g)': food1['Fiber (in g)'] + food2['Fiber (in g)'],
                    'Calcium (in mg)': food1['Calcium (in mg)'] + food2['Calcium (in mg)'],
                    'Total (in g)': food1['Total (in g)'] + food2['Total (in g)'],
                    'Calories (in Cal)': food1['Calories (in Cal)'] + food2['Calories (in Cal)'],
                    'GI': (food1['GI'] + food2['GI']) / 2,  # Average GI value
                    'Side': food2['Side'],
                    'Needside': food1['Needside'],
                    'rank_score': rank_score,
                    'meat_type':meat_type
                })
        
        # Save progress after each combination
        with open('progress.txt', 'w') as file:
            file.write(str(i + 1))
        
        # Optionally, save intermediate results to avoid data loss
        if (i + 1) % 100 == 0:
            save_ranked_combos(ranked_combos)
            ranked_combos = []  # Clear the list after saving

    return ranked_combos

# Function to save ranked combinations to CSV files
def save_ranked_combos(ranked_combos):
    if ranked_combos:
        ranked_df = pd.DataFrame(ranked_combos)
        for timing in ['b', 'l', 'd', 's']:
            timing_df = ranked_df[ranked_df['MealTimings'] == timing]
            if not timing_df.empty:
                timing_df.to_csv(f'telengana_{timing}.csv', mode='a', header=False, index=False)

# Save independent dishes directly to the respective CSV files
def save_independent_dishes(dishes):
    for timing in ['b', 'l', 'd', 's']:
        timing_df = dishes[dishes['MealTimings'].str.contains(timing)]
        if not timing_df.empty:
            timing_df.to_csv(f'telengana_{timing}.csv', mode='a', header=False, index=False)

# Save independent dishes
save_independent_dishes(independent_dishes)

# Manually rank the combinations
ranked_combos = manual_rank_combinations(combos, food_db, start_index)

# Save remaining ranked combinations
save_ranked_combos(ranked_combos)

print("Ranking and saving completed.")
