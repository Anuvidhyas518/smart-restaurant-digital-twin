import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/processed/restaurant_waste_processed.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. FEATURES
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
# 4. PREPROCESSOR
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
# 5. MODELS
# ==========================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Decision Tree":
        DecisionTreeRegressor(
            max_depth=4,
            random_state=42
        ),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
}


# ==========================================
# 6. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 7. MODEL COMPARISON
# ==========================================

results = []

for name, regressor in models.items():

    print()
    print("Training:", name)

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                regressor
            )
        ]
    )

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

results_df = pd.DataFrame(results)

print()
print("=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(
    results_df.to_string(index=False)
) 