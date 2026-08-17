import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/processed/restaurant_waste_processed.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. SELECT FEATURES
# ==========================================

features = [
    "customers",
    "day_of_week",
    "meal",
    "menu_type",
    "special_event",
    "temperature_c",
    "rainfall_mm"
]

target = "food_waste_kg"

X = df[features]
y = df[target]


# ==========================================
# 3. FEATURE TYPES
# ==========================================

categorical_features = [
    "meal",
    "menu_type",
    "special_event"
]

numeric_features = [
    "customers",
    "day_of_week",
    "temperature_c",
    "rainfall_mm"
]


# ==========================================
# 4. PREPROCESSING
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numeric",
            "passthrough",
            numeric_features
        )
    ]
)


# ==========================================
# 5. MODEL PIPELINE
# ==========================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        )
    ]
)


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 7. TRAIN MODEL
# ==========================================

model.fit(
    X_train,
    y_train
)

print()
print("Model training completed!")


# ==========================================
# 8. PREDICTION
# ==========================================

y_pred = model.predict(
    X_test
)

print()
print("Actual waste:")
print(y_test.values)

print()
print("Predicted waste:")
print(y_pred.round(2))


# ==========================================
# 9. EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)

print()
print("Model Performance")
print("-------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)


# ==========================================
# 10. SAVE MODEL
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

model_path = "models/waste_prediction_model.pkl"

joblib.dump(
    model,
    model_path
)

print()
print("Model saved successfully!")
print("Saved to:", model_path) 