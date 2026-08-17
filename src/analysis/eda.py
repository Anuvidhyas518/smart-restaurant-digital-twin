import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# LOAD PROCESSED DATA
# ==========================================

file_path = "data/processed/restaurant_waste_processed.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print()

# ==========================================
# CONVERT NUMERIC COLUMNS
# ==========================================

numeric_columns = [
    "customers",
    "food_prepared_kg",
    "food_served_kg",
    "food_waste_kg",
    "temperature_c",
    "rainfall_mm",
    "food_consumed_kg",
    "waste_percentage",
    "food_per_customer_kg"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

# ==========================================
# DATASET INFORMATION
# ==========================================

print("Dataset information:")
print(df.info())

print()

# ==========================================
# STATISTICAL SUMMARY
# ==========================================

print("Statistical summary:")
print(df.describe())

print()

# ==========================================
# AVERAGE WASTE BY MEAL
# ==========================================

meal_waste = (
    df.groupby("meal")["food_waste_kg"]
    .mean()
)

print("Average waste by meal:")
print(meal_waste)

# ==========================================
# GRAPH 1: WASTE BY MEAL
# ==========================================

plt.figure(figsize=(8, 5))

meal_waste.plot(kind="bar")

plt.title("Average Food Waste by Meal")
plt.xlabel("Meal")
plt.ylabel("Average Waste (kg)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ==========================================
# GRAPH 2: CUSTOMERS VS WASTE
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["customers"],
    df["food_waste_kg"]
)

plt.title("Customers vs Food Waste")
plt.xlabel("Number of Customers")
plt.ylabel("Food Waste (kg)")
plt.tight_layout()
plt.show()

# ==========================================
# GRAPH 3: FOOD PREPARED VS WASTE
# ==========================================

plt.figure(figsize=(8, 5))

plt.scatter(
    df["food_prepared_kg"],
    df["food_waste_kg"]
)

plt.title("Food Prepared vs Food Waste")
plt.xlabel("Food Prepared (kg)")
plt.ylabel("Food Waste (kg)")
plt.tight_layout()
plt.show()

# ==========================================
# GRAPH 4: WASTE PERCENTAGE BY MEAL
# ==========================================

meal_percentage = (
    df.groupby("meal")["waste_percentage"]
    .mean()
)

plt.figure(figsize=(8, 5))

meal_percentage.plot(kind="bar")

plt.title("Average Waste Percentage by Meal")
plt.xlabel("Meal")
plt.ylabel("Waste Percentage (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# ==========================================
# GRAPH 5: WASTE BY MENU TYPE
# ==========================================

menu_waste = (
    df.groupby("menu_type")["food_waste_kg"]
    .mean()
)

print()
print("Average waste by menu type:")
print(menu_waste)

plt.figure(figsize=(8, 5))

menu_waste.plot(kind="bar")

plt.title("Average Food Waste by Menu Type")
plt.xlabel("Menu Type")
plt.ylabel("Average Waste (kg)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

print()
print("EDA completed successfully!")
