import joblib
import pandas as pd

# Load trained model
model_file = "models/waste_prediction_model.pkl"

model = joblib.load(model_file)

print("AI model loaded successfully!")


# --------------------------------
# New restaurant information
# --------------------------------

customers = 120
food_prepared_kg = 30
food_served_kg = 27
day_of_week = 2


# Create input data
new_data = pd.DataFrame({
    "customers": [customers],
    "food_prepared_kg": [food_prepared_kg],
    "food_served_kg": [food_served_kg],
    "day_of_week": [day_of_week]
})


# --------------------------------
# Predict food waste
# --------------------------------

prediction = model.predict(new_data)

predicted_waste = prediction[0]


print("\n--------------------------------")
print("RESTAURANT WASTE PREDICTION")
print("--------------------------------")

print("Customers:", customers)
print("Food prepared:", food_prepared_kg, "kg")
print("Food served:", food_served_kg, "kg")
print("Day of week:", day_of_week)

print("\nPredicted food waste:",
      round(predicted_waste, 2), "kg") 
