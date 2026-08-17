import pandas as pd
import os

# ==========================================
# FILE PATHS
# ==========================================

input_file = "data/raw/restaurant_data_collection.csv"
output_file = "data/processed/restaurant_waste_processed.csv"


# ==========================================
# CREATE OUTPUT FOLDER
# ==========================================

os.makedirs("data/processed", exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(input_file)

print("Original data:")
print(df)


# ==========================================
# REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()


# ==========================================
# CONVERT DATE
# ==========================================

df["date"] = pd.to_datetime(df["date"])

df["day"] = df["date"].dt.day_name()

df["day_of_week"] = df["date"].dt.dayofweek


# ==========================================
# CONVERT NUMERIC COLUMNS
# ==========================================

numeric_columns = [
    "customers",
    "food_prepared_kg",
    "food_served_kg",
    "food_waste_kg",
    "temperature_c",
    "rainfall_mm"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ==========================================
# REMOVE INVALID VALUES
# ==========================================

df = df[
    (df["customers"] > 0) &
    (df["food_prepared_kg"] >= 0) &
    (df["food_served_kg"] >= 0) &
    (df["food_waste_kg"] >= 0)
]


# ==========================================
# CALCULATE FOOD CONSUMED
# ==========================================

df["food_consumed_kg"] = (
    df["food_served_kg"] -
    df["food_waste_kg"]
)


# ==========================================
# CALCULATE WASTE PERCENTAGE
# ==========================================

df["waste_percentage"] = (
    df["food_waste_kg"] /
    df["food_prepared_kg"]
) * 100


# ==========================================
# CALCULATE FOOD PER CUSTOMER
# ==========================================

df["food_per_customer_kg"] = (
    df["food_prepared_kg"] /
    df["customers"]
)


# ==========================================
# REMOVE INVALID CALCULATIONS
# ==========================================

df = df[
    (df["food_consumed_kg"] >= 0) &
    (df["waste_percentage"] >= 0)
]


# ==========================================
# DISPLAY PROCESSED DATA
# ==========================================

print("\nProcessed data:")
print(df)


# ==========================================
# SAVE PROCESSED DATA
# ==========================================

df.to_csv(
    output_file,
    index=False
)

print("\nProcessed data saved successfully!")
print("File saved to:", output_file) 