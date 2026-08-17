import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = "data/processed/restaurant_waste_processed.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 2. FEATURES AND TARGET
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
# 3. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 4. LOAD TRAINED MODEL
# ==========================================

model = joblib.load(
    "models/waste_prediction_model.pkl"
)


# ==========================================
# 5. PERMUTATION IMPORTANCE
# ==========================================

result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error"
)


# ==========================================
# 6. CREATE RESULTS TABLE
# ==========================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": result.importances_mean
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)


# ==========================================
# 7. DISPLAY
# ==========================================

print()
print("==========================================")
print("FEATURE IMPORTANCE")
print("==========================================")

print(
    importance.to_string(index=False)
) 