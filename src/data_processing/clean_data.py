import pandas as pd
import os

# File paths
input_file = "data/raw/restaurant_data_collection.csv"
output_file = "data/processed/restaurant_waste_processed.csv"

# Load raw data
df = pd.read_csv(input_file)

print("RAW DATA")
print("=" * 50)
print(df)
print("\nColumns:", df.columns.tolist())
print("Total records:", len(df))

# Rename the actual column to the standard name used by the project
df = df.rename(columns={
    "food_waste_kg": "food_wasted_kg"
})
# Clean hotel names
df["hotel_name"] = (
    df["hotel_name"]
    .astype(str)
    .str.strip()
)

# Standardize restaurant names
df["hotel_name"] = df["hotel_name"].replace({
    "Bhai Briyani": "Bhai Biriyani"
})


# Required columns
required_columns = [
    "day",
    "hotel_name",
    "food_prepared_kg",
    "food_wasted_kg"
]

# Check columns
missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

# Convert numerical columns
df["day"] = pd.to_numeric(df["day"], errors="coerce")
df["food_prepared_kg"] = pd.to_numeric(
    df["food_prepared_kg"], errors="coerce"
)
df["food_wasted_kg"] = pd.to_numeric(
    df["food_wasted_kg"], errors="coerce"
)

# Remove invalid rows
df = df.dropna(
    subset=[
        "day",
        "hotel_name",
        "food_prepared_kg",
        "food_wasted_kg"
    ]
)

# Day of week
df["day_of_week"] = ((df["day"] - 1) % 7) + 1

# Waste percentage
df["waste_percentage"] = (
    df["food_wasted_kg"] /
    df["food_prepared_kg"]
) * 100

# Create output directory
os.makedirs(
    os.path.dirname(output_file),
    exist_ok=True
)

# Save processed data
df.to_csv(output_file, index=False)

print("\nCLEANING COMPLETED")
print("=" * 50)
print("Valid records:", len(df))
print("\nProcessed data:")
print(
    df[
        [
            "day",
            "hotel_name",
            "food_prepared_kg",
            "food_wasted_kg",
            "day_of_week",
            "waste_percentage"
        ]
    ].head(20)
)

print("\nSaved successfully:")
print(output_file) 