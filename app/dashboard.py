import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


# ==========================================
# PROJECT PATH
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================
# WASTE UTILIZATION IMPORT
# ==========================================

from src.waste_utilization.waste_output_estimator import (
    estimate_utilization
) 
from src.waste_utilization.impact_calculator import (
    calculate_period_impact
) 
# ==========================================
# LOAD AI MODEL
# ==========================================

model = joblib.load(
    "models/waste_prediction_model.pkl"
)


# ==========================================
# LOAD HISTORICAL DATA
# ==========================================

historical_data = pd.read_csv(
    "data/processed/restaurant_waste_processed.csv"
)

historical_data["date"] = pd.to_datetime(
    historical_data["date"]
)


# ==========================================
# PAGE TITLE
# ==========================================

st.title("🍽️ Smart Restaurant Digital Twin")

st.write(
    "AI-powered food waste prediction and "
    "restaurant waste management system."
)


# ==========================================
# WASTE PREDICTION
# ==========================================

st.header("🔮 Food Waste Prediction")

customers = st.number_input(
    "Number of Customers",
    min_value=1,
    max_value=1000,
    value=100
)


meal = st.selectbox(
    "Meal",
    [
        "Lunch",
        "Dinner"
    ]
)


menu_type = st.selectbox(
    "Menu Type",
    [
        "South Indian",
        "North Indian",
        "Chinese",
        "Fast Food",
        "Mixed"
    ]
)


special_event = st.selectbox(
    "Special Event?",
    [
        "No",
        "Yes"
    ]
)


temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=30.0
)


rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=0.0
)


day_of_week = st.selectbox(
    "Day of Week",
    [
        ("Monday", 0),
        ("Tuesday", 1),
        ("Wednesday", 2),
        ("Thursday", 3),
        ("Friday", 4),
        ("Saturday", 5),
        ("Sunday", 6)
    ],
    format_func=lambda x: x[0]
)


# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("🔮 Predict Food Waste"):

    prediction_input = pd.DataFrame({
        "customers": [customers],
        "day_of_week": [day_of_week[1]],
        "meal": [meal],
        "menu_type": [menu_type],
        "special_event": [special_event],
        "temperature_c": [temperature],
        "rainfall_mm": [rainfall]
    })

    predicted_waste = model.predict(
        prediction_input
    )[0]

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Food Waste: "
        f"{predicted_waste:.2f} kg"
    )

    # ==========================================
    # WASTE RISK
    # ==========================================

    if predicted_waste > 4:

        st.error(
            "🔴 HIGH WASTE RISK"
        )

        st.write(
            "Recommendation: Reduce preparation based "
            "on expected demand and monitor leftovers."
        )

    elif predicted_waste > 2:

        st.warning(
            "🟡 MODERATE WASTE RISK"
        )

        st.write(
            "Recommendation: Monitor demand and "
            "adjust preparation carefully."
        )

    else:

        st.success(
            "🟢 LOW WASTE RISK"
        )

        st.write(
            "Recommendation: Current operating conditions "
            "show relatively low predicted waste."
        )


# ==========================================
# HISTORICAL FOOD WASTE ANALYSIS
# ==========================================

st.header("📊 Historical Food Waste Analysis")


average_waste = historical_data[
    "food_waste_kg"
].mean()

maximum_waste = historical_data[
    "food_waste_kg"
].max()

total_waste = historical_data[
    "food_waste_kg"
].sum()


col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Waste",
    f"{average_waste:.2f} kg"
)

col2.metric(
    "Maximum Waste",
    f"{maximum_waste:.2f} kg"
)

col3.metric(
    "Total Waste",
    f"{total_waste:.2f} kg"
)


# ==========================================
# WASTE TREND
# ==========================================

st.subheader("📈 Food Waste Trend")

fig1, ax1 = plt.subplots()

ax1.plot(
    historical_data["date"],
    historical_data["food_waste_kg"],
    marker="o"
)

ax1.set_xlabel("Date")
ax1.set_ylabel("Food Waste (kg)")
ax1.set_title("Food Waste Over Time")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig1)


# ==========================================
# CUSTOMERS VS WASTE
# ==========================================

st.subheader("👥 Customers vs Food Waste")

fig2, ax2 = plt.subplots()

ax2.scatter(
    historical_data["customers"],
    historical_data["food_waste_kg"]
)

ax2.set_xlabel("Customers")
ax2.set_ylabel("Food Waste (kg)")
ax2.set_title("Customers vs Food Waste")

plt.tight_layout()

st.pyplot(fig2)


# ==========================================
# MENU TYPE ANALYSIS
# ==========================================

st.subheader("🍛 Waste by Menu Type")

menu_waste = historical_data.groupby(
    "menu_type"
)["food_waste_kg"].mean()

fig3, ax3 = plt.subplots()

menu_waste.plot(
    kind="bar",
    ax=ax3
)

ax3.set_xlabel("Menu Type")
ax3.set_ylabel("Average Waste (kg)")
ax3.set_title("Average Waste by Menu Type")

plt.xticks(rotation=45)
plt.tight_layout()

st.pyplot(fig3)


# ==========================================
# WHAT-IF SIMULATION
# ==========================================

st.header("🔄 What-If Simulation")

st.write(
    "Change operating conditions to estimate "
    "future food waste before food preparation."
)


simulation_customers = st.slider(
    "Number of Customers",
    min_value=50,
    max_value=300,
    value=120,
    step=10
)


simulation_meal = st.selectbox(
    "Simulation Meal",
    [
        "Lunch",
        "Dinner"
    ],
    key="simulation_meal"
)


simulation_menu = st.selectbox(
    "Simulation Menu Type",
    [
        "South Indian",
        "North Indian",
        "Chinese",
        "Fast Food",
        "Mixed"
    ],
    key="simulation_menu"
)


simulation_event = st.selectbox(
    "Simulation Special Event",
    [
        "No",
        "Yes"
    ],
    key="simulation_event"
)


simulation_temperature = st.number_input(
    "Simulation Temperature (°C)",
    min_value=0.0,
    max_value=50.0,
    value=30.0,
    key="simulation_temperature"
)


simulation_rainfall = st.number_input(
    "Simulation Rainfall (mm)",
    min_value=0.0,
    max_value=500.0,
    value=0.0,
    key="simulation_rainfall"
)


simulation_day = st.selectbox(
    "Simulation Day",
    [
        ("Monday", 0),
        ("Tuesday", 1),
        ("Wednesday", 2),
        ("Thursday", 3),
        ("Friday", 4),
        ("Saturday", 5),
        ("Sunday", 6)
    ],
    format_func=lambda x: x[0],
    key="simulation_day"
)
# ==========================================
# PREPARATION RECOMMENDATION
# ==========================================

st.header("🎯 Preparation Recommendation")

matching_data = historical_data[
    (historical_data["meal"] == simulation_meal) &
    (historical_data["menu_type"] == simulation_menu)
]

if len(matching_data) == 0:
    matching_data = historical_data

historical_rate = (
    matching_data["food_prepared_kg"] /
    matching_data["customers"]
).mean()

recommended_preparation = (
    simulation_customers *
    historical_rate
)

st.success(
    f"Recommended Food Preparation: "
    f"{recommended_preparation:.2f} kg"
)

st.info(
    f"Historical preparation rate: "
    f"{historical_rate:.3f} kg/customer"
) 

# ==========================================
# SIMULATION INPUT
# ==========================================

simulation_data = pd.DataFrame({
    "customers": [simulation_customers],
    "day_of_week": [simulation_day[1]],
    "meal": [simulation_meal],
    "menu_type": [simulation_menu],
    "special_event": [simulation_event],
    "temperature_c": [simulation_temperature],
    "rainfall_mm": [simulation_rainfall]
})


# ==========================================
# SIMULATION PREDICTION
# ==========================================

simulation_waste = model.predict(
    simulation_data
)[0]


st.subheader("Simulation Result")

st.metric(
    "Predicted Food Waste",
    f"{simulation_waste:.2f} kg"
)


# ==========================================
# SIMULATION RISK
# ==========================================

if simulation_waste > 4:

    st.error(
        "🔴 High waste predicted."
    )

elif simulation_waste > 2:

    st.warning(
        "🟡 Moderate waste predicted."
    )

else:

    st.success(
        "🟢 Low waste predicted."
    )


# ==========================================
# PREPARATION RECOMMENDATION
# ==========================================

st.header("🎯 Preparation Recommendation")

st.write(
    "Because the current AI model predicts waste from "
    "pre-preparation conditions, preparation quantity is "
    "estimated separately from historical food-per-customer "
    "behavior."
)


# Filter historical data using meal and menu type
matching_data = historical_data[
    (historical_data["meal"] == simulation_meal) &
    (historical_data["menu_type"] == simulation_menu)
]


if len(matching_data) == 0:

    matching_data = historical_data


# Calculate historical food prepared per customer

historical_rate = (
    matching_data["food_prepared_kg"] /
    matching_data["customers"]
).mean()


recommended_preparation = (
    simulation_customers *
    historical_rate
)


st.success(
    f"Recommended Food Preparation: "
    f"{recommended_preparation:.2f} kg"
)


st.info(
    f"Historical preparation rate: "
    f"{historical_rate:.3f} kg/customer"
)


# ==========================================
# WASTE MANAGEMENT RECOMMENDATION
# ==========================================

st.header("♻️ Waste Management Recommendation")


st.metric(
    "Predicted Waste for Management",
    f"{simulation_waste:.2f} kg"
)


if simulation_waste > 4:

    st.error(
        "🔴 HIGH WASTE LEVEL"
    )

    st.write(
        "Recommended actions:"
    )

    st.write(
        "• Reduce excess preparation."
    )

    st.write(
        "• Monitor expected customer demand."
    )

    st.write(
        "• Review menu-level waste patterns."
    )

elif simulation_waste > 2:

    st.warning(
        "🟡 MODERATE WASTE LEVEL"
    )

    st.write(
        "Recommended actions:"
    )

    st.write(
        "• Adjust preparation according to demand."
    )

    st.write(
        "• Monitor leftover food."
    )

    st.write(
        "• Review menu-specific waste."
    )

else:

    st.success(
        "🟢 LOW WASTE LEVEL"
    )

    st.write(
        "Current operating conditions show "
        "relatively low predicted waste."
    )
# ==========================================
# AI MODEL EXPLAINABILITY
# ==========================================

st.header("🧠 AI Model Explainability")

st.write(
    "This section shows which input factors have "
    "the greatest influence on the model's predictions."
)


# Features used by the model
explain_features = [
    "customers",
    "day_of_week",
    "meal",
    "menu_type",
    "special_event",
    "temperature_c",
    "rainfall_mm"
]

X_explain = historical_data[explain_features]

y_explain = historical_data["food_waste_kg"]


# Split data
X_train_explain, X_test_explain, y_train_explain, y_test_explain = (
    train_test_split(
        X_explain,
        y_explain,
        test_size=0.20,
        random_state=42
    )
)


# Calculate permutation importance
importance_result = permutation_importance(
    model,
    X_test_explain,
    y_test_explain,
    n_repeats=10,
    random_state=42,
    scoring="neg_mean_absolute_error"
)


# Create importance table
importance_df = pd.DataFrame({
    "Feature": explain_features,
    "Importance": importance_result.importances_mean
})


importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


# Display table
st.subheader("Feature Importance")

st.dataframe(
    importance_df,
    use_container_width=True
)


# Display graph
st.subheader("Factors Influencing Waste Prediction")

fig4, ax4 = plt.subplots()

ax4.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax4.set_xlabel(
    "Permutation Importance"
)

ax4.set_ylabel(
    "Feature"
)

ax4.set_title(
    "AI Model Feature Importance"
)

ax4.invert_yaxis()

plt.tight_layout()

st.pyplot(fig4)

# ==========================================
# REAL RESTAURANT DATA ENTRY
# ==========================================

st.header("📝 Add Restaurant Observation")

st.write(
    "Enter an actual restaurant observation "
    "to add it to the data collection file."
)

with st.form("restaurant_data_form"):

    entry_date = st.date_input(
        "Date"
    )

    entry_meal = st.selectbox(
        "Meal",
        [
            "Lunch",
            "Dinner"
        ],
        key="entry_meal"
    )

    entry_customers = st.number_input(
        "Customers",
        min_value=1,
        max_value=1000,
        value=100,
        step=1,
        key="entry_customers"
    )

    entry_prepared = st.number_input(
        "Food Prepared (kg)",
        min_value=0.0,
        max_value=500.0,
        value=25.0,
        step=0.5,
        key="entry_prepared"
    )

    entry_served = st.number_input(
        "Food Served (kg)",
        min_value=0.0,
        max_value=500.0,
        value=22.0,
        step=0.5,
        key="entry_served"
    )

    entry_waste = st.number_input(
        "Food Waste (kg)",
        min_value=0.0,
        max_value=500.0,
        value=3.0,
        step=0.5,
        key="entry_waste"
    )

    entry_menu = st.selectbox(
        "Menu Type",
        [
            "South Indian",
            "North Indian",
            "Chinese",
            "Fast Food",
            "Mixed"
        ],
        key="entry_menu"
    )

    entry_event = st.selectbox(
        "Special Event?",
        [
            "No",
            "Yes"
        ],
        key="entry_event"
    )

    entry_temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=30.0,
        step=0.5,
        key="entry_temperature"
    )

    entry_rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=0.0,
        step=0.5,
        key="entry_rainfall"
    )

    entry_category = st.selectbox(
        "Waste Category",
        [
            "Edible Surplus",
            "Vegetable/Fruit Waste",
            "Cooked Food Waste",
            "Inedible Organic Waste",
            "Packaging Waste"
        ],
        key="entry_category"
    )

    submitted = st.form_submit_button(
        "💾 Save Restaurant Observation"
    )


# ==========================================
# SAVE OBSERVATION
# ==========================================

if submitted:

    errors = []

    # ==========================================
    # VALIDATE INPUT
    # ==========================================

    if entry_customers <= 0:
        errors.append("Customers must be greater than 0.")

    if entry_prepared < 0:
        errors.append("Food prepared cannot be negative.")

    if entry_served < 0:
        errors.append("Food served cannot be negative.")

    if entry_waste < 0:
        errors.append("Food waste cannot be negative.")

    if entry_served > entry_prepared:
        errors.append(
            "Food served cannot be greater than food prepared."
        )

    if entry_waste > entry_prepared:
        errors.append(
            "Food waste cannot be greater than food prepared."
        )

    if entry_temperature < 0:
        errors.append(
            "Temperature cannot be negative."
        )

    if entry_rainfall < 0:
        errors.append(
            "Rainfall cannot be negative."
        )


    # ==========================================
    # SHOW ERRORS
    # ==========================================

    if errors:

        st.error(
            "Please correct the following:"
        )

        for error in errors:
            st.write(f"• {error}")


    # ==========================================
    # SAVE VALID RECORD
    # ==========================================

    else:

        new_record = pd.DataFrame({
            "date": [
                entry_date.strftime("%Y-%m-%d")
            ],
            "day": [
                entry_date.strftime("%A")
            ],
            "meal": [
                entry_meal
            ],
            "customers": [
                entry_customers
            ],
            "food_prepared_kg": [
                entry_prepared
            ],
            "food_served_kg": [
                entry_served
            ],
            "food_waste_kg": [
                entry_waste
            ],
            "menu_type": [
                entry_menu
            ],
            "special_event": [
                entry_event
            ],
            "temperature_c": [
                entry_temperature
            ],
            "rainfall_mm": [
                entry_rainfall
            ],
            "waste_category": [
                entry_category
            ]
        })


        raw_file = (
            "data/raw/restaurant_data_collection.csv"
        )


        try:

            existing_data = pd.read_csv(
                raw_file
            )

            updated_data = pd.concat(
                [
                    existing_data,
                    new_record
                ],
                ignore_index=True
            )

        except FileNotFoundError:

            updated_data = new_record


        updated_data.to_csv(
            raw_file,
            index=False
        )


        st.success(
            "✅ Restaurant observation saved successfully!"
        )

        st.write(
            f"Total observations: "
            f"{len(updated_data)}"
        ) 
# ==========================================
# WASTE UTILIZATION MODULE
# ==========================================

st.header("♻️ Waste Utilization Recommendation")

st.write(
    "Select the waste type to receive an "
    "appropriate management recommendation."
)


waste_quantity = simulation_waste


st.metric(
    "AI Predicted Waste",
    f"{waste_quantity:.2f} kg"
)


waste_type = st.selectbox(
    "Waste Type",
    [
        "Edible Surplus",
        "Vegetable / Fruit Waste",
        "Cooked Food Waste",
        "Inedible Organic Waste",
        "Packaging Waste"
    ]
)


if st.button(
    "♻️ Recommend Waste Utilization"
):

    if waste_type == "Edible Surplus":

        recommendation = (
            "Safe redistribution, where permitted"
        )

        explanation = (
            "Only food that has been safely handled "
            "and remains suitable for consumption "
            "should be considered for redistribution."
        )


    elif waste_type == "Vegetable / Fruit Waste":

        recommendation = "Composting"

        explanation = (
            "Vegetable and fruit residues can be "
            "directed to controlled composting."
        )


    elif waste_type == "Cooked Food Waste":

        recommendation = (
            "Composting / Organic Waste Processing"
        )

        explanation = (
            "Cooked food waste can be directed to "
            "an appropriate organic-waste processing "
            "system."
        )


    elif waste_type == "Inedible Organic Waste":

        recommendation = (
            "Composting / Biogas Processing"
        )

        explanation = (
            "Inedible organic waste may be suitable "
            "for composting or anaerobic digestion "
            "where available."
        )


    else:

        recommendation = (
            "Recycling / Appropriate Disposal"
        )

        explanation = (
            "Packaging should be separated from food "
            "waste and sent to the appropriate recycling "
            "or disposal stream."
        )


    st.subheader(
        "Recommended Action"
    )

    st.success(
        f"♻️ {recommendation}"
    )

    st.info(
        explanation
    )

    st.metric(
        "Waste Quantity",
        f"{waste_quantity:.2f} kg"
    ) 
# ==========================================
# WASTE-TO-RESOURCE POTENTIAL
# ==========================================

st.header("⚡ Waste-to-Resource Potential")

st.write(
    "Estimate the potential amount of useful resources "
    "that could be obtained from the predicted food waste."
)


# ------------------------------------------
# Waste quantity
# ------------------------------------------

utilization_waste = st.number_input(
    "Waste Available for Utilization (kg)",
    min_value=0.0,
    max_value=500.0,
    value=float(round(simulation_waste, 2)),
    step=0.5,
    key="utilization_waste"
)


# ------------------------------------------
# Scenario assumptions
# ------------------------------------------

organic_fraction = st.slider(
    "Organic Fraction",
    min_value=0.0,
    max_value=1.0,
    value=1.0,
    step=0.05,
    key="organic_fraction"
)

volatile_solids_fraction = st.slider(
    "Volatile Solids Fraction",
    min_value=0.01,
    max_value=0.80,
    value=0.20,
    step=0.01,
    key="volatile_solids_fraction"
)

methane_fraction = st.slider(
    "Methane Fraction of Biogas",
    min_value=0.40,
    max_value=0.80,
    value=0.60,
    step=0.01,
    key="methane_fraction"
)

compost_recovery_rate = st.slider(
    "Compost Recovery Scenario",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05,
    key="compost_recovery_rate"
)


# ------------------------------------------
# Calculate
# ------------------------------------------

if st.button(
    "⚡ Estimate Resource Potential"
):

    utilization = estimate_utilization(
        waste_kg=utilization_waste,
        organic_fraction=organic_fraction,
        volatile_solids_fraction=volatile_solids_fraction,
        methane_fraction=methane_fraction,
        compost_recovery_rate=compost_recovery_rate
    )


    # ------------------------------------------
    # General outputs
    # ------------------------------------------

    st.subheader("Estimated Resource Outputs")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Organic Waste",
        f"{utilization['organic_waste_kg']:.2f} kg"
    )

    col2.metric(
        "Volatile Solids",
        f"{utilization['volatile_solids_kg']:.2f} kg"
    )

    col3.metric(
        "Compost Potential",
        f"{utilization['compost_potential_kg']:.2f} kg"
    )


    # ------------------------------------------
    # Biogas
    # ------------------------------------------

    st.subheader("🔥 Biogas Potential")

    col4, col5 = st.columns(2)

    col4.metric(
        "Methane Potential",
        (
            f"{utilization['methane_low_m3']:.2f} – "
            f"{utilization['methane_high_m3']:.2f} m³ CH₄"
        )
    )

    col5.metric(
        "Biogas Potential",
        (
            f"{utilization['biogas_low_m3']:.2f} – "
            f"{utilization['biogas_high_m3']:.2f} m³"
        )
    )


    st.info(
        "These are theoretical/scenario estimates, not "
        "measured biogas or compost production. Actual "
        "outputs depend on waste composition and processing "
        "conditions."
    ) 
# ==========================================
# RESTAURANT MONITORING SUMMARY
# ==========================================

st.header("📋 Restaurant Monitoring Summary")

try:

    monitoring_data = pd.read_csv(
        "data/raw/restaurant_data_collection.csv"
    )

    if len(monitoring_data) == 0:

        st.info(
            "No restaurant observations have been collected yet."
        )

    else:

        monitoring_data["food_waste_kg"] = pd.to_numeric(
            monitoring_data["food_waste_kg"],
            errors="coerce"
        )

        monitoring_data["customers"] = pd.to_numeric(
            monitoring_data["customers"],
            errors="coerce"
        )

        # ------------------------------------------
        # SUMMARY VALUES
        # ------------------------------------------

        total_observations = len(
            monitoring_data
        )

        total_waste = (
            monitoring_data["food_waste_kg"]
            .sum()
        )

        average_waste = (
            monitoring_data["food_waste_kg"]
            .mean()
        )

        average_customers = (
            monitoring_data["customers"]
            .mean()
        )

        # ------------------------------------------
        # DISPLAY SUMMARY
        # ------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Observations",
            total_observations
        )

        col2.metric(
            "Total Waste",
            f"{total_waste:.2f} kg"
        )

        col3.metric(
            "Average Waste",
            f"{average_waste:.2f} kg"
        )

        col4.metric(
            "Average Customers",
            f"{average_customers:.0f}"
        )

        # ------------------------------------------
        # HIGHEST WASTE MEAL
        # ------------------------------------------

        meal_summary = (
            monitoring_data
            .groupby("meal")["food_waste_kg"]
            .mean()
            .sort_values(ascending=False)
        )

        if not meal_summary.empty:

            highest_waste_meal = (
                meal_summary.index[0]
            )

            highest_waste_meal_value = (
                meal_summary.iloc[0]
            )

            st.info(
                f"Highest average-waste meal: "
                f"{highest_waste_meal} "
                f"({highest_waste_meal_value:.2f} kg)"
            )

        # ------------------------------------------
        # HIGHEST WASTE MENU
        # ------------------------------------------

        menu_summary = (
            monitoring_data
            .groupby("menu_type")["food_waste_kg"]
            .mean()
            .sort_values(ascending=False)
        )

        if not menu_summary.empty:

            highest_waste_menu = (
                menu_summary.index[0]
            )

            highest_waste_menu_value = (
                menu_summary.iloc[0]
            )

            st.info(
                f"Highest average-waste menu: "
                f"{highest_waste_menu} "
                f"({highest_waste_menu_value:.2f} kg)"
            )

        # ------------------------------------------
        # RECENT OBSERVATIONS
        # ------------------------------------------

        st.subheader(
            "Recent Restaurant Observations"
        )

        st.dataframe(
            monitoring_data.tail(10),
            use_container_width=True
        )

except FileNotFoundError:

    st.warning(
        "Restaurant data collection file was not found."
    ) 
# ==========================================
# MODEL RETRAINING
# ==========================================

st.header("🔄 Model Retraining")

st.write(
    "After collecting a sufficient batch of new "
    "restaurant observations, retrain the AI model."
)

retrain = st.button(
    "🔄 Retrain AI Model"
)

if retrain:

    import subprocess
    import sys

    with st.spinner(
        "Cleaning data and retraining the model..."
    ):

        result = subprocess.run(
            [
                sys.executable,
                "src/prediction/retrain_model.py"
            ],
            capture_output=True,
            text=True
        )

    if result.returncode == 0:

        st.success(
            "✅ AI model retrained successfully!"
        )

        st.code(
            result.stdout
        )

        st.info(
            "Restart or refresh the dashboard to "
            "use the updated model."
        )

    else:

        st.error(
            "❌ Model retraining failed."
        )

        st.code(
            result.stderr
        ) 

# ==========================================
# WEEKLY / MONTHLY IMPACT
# ==========================================

st.header("📅 Weekly / Monthly Impact")

impact_data = pd.read_csv(
    "data/raw/restaurant_data_collection.csv"
)

impact_data["date"] = pd.to_datetime(
    impact_data["date"]
)

impact_data["food_waste_kg"] = pd.to_numeric(
    impact_data["food_waste_kg"],
    errors="coerce"
)

impact_data = impact_data.dropna(
    subset=["date", "food_waste_kg"]
)

period_type = st.selectbox(
    "Impact Period",
    ["Weekly", "Monthly"],
    key="impact_period_type"
)

if period_type == "Weekly":

    impact_data["period"] = (
        impact_data["date"]
        .dt.to_period("W")
        .astype(str)
    )

else:

    impact_data["period"] = (
        impact_data["date"]
        .dt.to_period("M")
        .astype(str)
    )


period_summary = (
    impact_data
    .groupby("period")["food_waste_kg"]
    .sum()
    .reset_index()
)


if period_summary.empty:

    st.info(
        "Not enough restaurant data for period analysis."
    )

else:

    selected_period = st.selectbox(
        "Select Period",
        period_summary["period"].tolist(),
        key="selected_impact_period"
    )

    selected_waste = period_summary.loc[
        period_summary["period"] == selected_period,
        "food_waste_kg"
    ].iloc[0]


    st.subheader(
        f"Impact for {selected_period}"
    )


    disposal_cost = st.number_input(
        "Waste Disposal Cost (₹/kg)",
        min_value=0.0,
        value=0.0,
        step=0.5,
        key="disposal_cost"
    )

    energy_value = st.number_input(
        "Energy Value (₹/kWh)",
        min_value=0.0,
        value=8.0,
        step=0.5,
        key="energy_value"
    )


    impact = calculate_period_impact(
        waste_kg=float(selected_waste),
        disposal_cost_per_kg=float(
            disposal_cost
        ),
        energy_value_per_kwh=float(
            energy_value
        )
    )


    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Food Waste",
        f"{impact['waste_kg']:.2f} kg"
    )

    col2.metric(
        "Biogas Potential",
        (
            f"{impact['biogas_low_m3']:.2f} – "
            f"{impact['biogas_high_m3']:.2f} m³"
        )
    )

    col3.metric(
        "Energy Potential",
        (
            f"{impact['energy_low_kwh']:.2f} – "
            f"{impact['energy_high_kwh']:.2f} kWh"
        )
    )


    st.subheader("💰 Potential Economic Value")


    col4, col5, col6 = st.columns(3)

    col4.metric(
        "Energy Value",
        (
            f"₹{impact['energy_value_low']:.2f} – "
            f"₹{impact['energy_value_high']:.2f}"
        )
    )

    col5.metric(
        "Avoided Disposal Cost",
        f"₹{impact['disposal_saving']:.2f}"
    )

    col6.metric(
        "Total Potential Value",
        (
            f"₹{impact['total_value_low']:.2f} – "
            f"₹{impact['total_value_high']:.2f}"
        )
    )


    st.subheader("🔥 Methane Potential")

    st.write(
        f"{impact['methane_low_m3']:.2f} – "
        f"{impact['methane_high_m3']:.2f} m³ CH₄"
    )


    st.info(
        "Biogas, methane, energy and monetary figures are "
        "potential estimates based on the selected assumptions. "
        "They are not measured production or guaranteed savings."
    ) 