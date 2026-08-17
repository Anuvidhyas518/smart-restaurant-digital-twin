import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# ==========================================
# LOAD DATA
# ==========================================

file_path = "data/processed/restaurant_waste_processed.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# FEATURES
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

X = df[features]
y = df["food_waste_kg"]


# ==========================================
# FEATURE TYPES
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
# PREPROCESSOR
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
# MODEL
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
# CROSS-VALIDATION
# ==========================================

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scores = cross_validate(
    model,
    X,
    y,
    cv=kfold,
    scoring={
        "MAE": "neg_mean_absolute_error",
        "RMSE": "neg_root_mean_squared_error",
        "R2": "r2"
    }
)


# ==========================================
# RESULTS
# ==========================================

mae_scores = -scores["test_MAE"]
rmse_scores = -scores["test_RMSE"]
r2_scores = scores["test_R2"]


print()
print("==========================================")
print("5-FOLD CROSS-VALIDATION RESULTS")
print("==========================================")

print("MAE scores:", mae_scores)
print("RMSE scores:", rmse_scores)
print("R2 scores:", r2_scores)

print()
print("Average MAE:", mae_scores.mean())
print("Average RMSE:", rmse_scores.mean())
print("Average R2:", r2_scores.mean()) 