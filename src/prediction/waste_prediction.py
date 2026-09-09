
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# LOAD PROCESSED DATA
# ==========================================

data_file = "data/processed/restaurant_waste_processed.csv"

df = pd.read_csv(data_file)

print("Processed data loaded successfully.")
print("Dataset shape:", df.shape)


# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "day",
    "hotel_name",
    "food_prepared_kg",
    "food_wasted_kg"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# ==========================================
# CONVERT HOTEL NAME TO NUMERIC
# ==========================================

df["hotel_code"] = (
    df["hotel_name"]
    .astype("category")
    .cat.codes
)


# ==========================================
# FEATURES
# ==========================================

X = df[
    [
        "day",
        "hotel_code",
        "food_prepared_kg"
    ]
]


# ==========================================
# TARGET
# ==========================================

y = df["food_wasted_kg"]


# ==========================================
# CHECK DATASET SIZE
# ==========================================

if len(df) < 10:
    print()
    print(
        "WARNING: Only",
        len(df),
        "records are available."
    )
    print(
        "More data is recommended for reliable "
        "AI prediction."
    )


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# RANDOM FOREST MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)


# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# MODEL METRICS
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    y_pred
)


# ==========================================
# DISPLAY RESULTS
# ==========================================

print()
print("==========================================")
print("MODEL TRAINING COMPLETED")
print("==========================================")

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print()
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 2))


# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print()
print("Feature Importance:")
print("------------------------------------------")

for feature, importance in zip(
    X.columns,
    model.feature_importances_
):
    print(
        feature,
        ":",
        round(importance, 3)
    )


# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model_file = "models/waste_prediction_model.pkl"

joblib.dump(
    model,
    model_file
)

print()
print("Model saved successfully.")
print("File:", model_file)

