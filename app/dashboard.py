import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Restaurant Digital Twin",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Smart Restaurant Digital Twin")
st.write(
    "AI-powered food waste prediction, hotel monitoring, "
    "digital twin simulation and waste reduction"
)

# --------------------------------------------------
# LOAD DATA AND MODEL
# --------------------------------------------------

data_file = "data/processed/restaurant_waste_processed.csv"
model_file = "models/waste_prediction_model.pkl"

try:
    df = pd.read_csv(data_file)
    model = joblib.load(model_file)
except Exception as e:
    st.error(f"Unable to load data or model: {e}")
    st.stop()

# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "day",
    "hotel_name",
    "food_prepared_kg",
    "food_wasted_kg"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        f"Missing columns in dataset: {missing_columns}"
    )
    st.stop()

# --------------------------------------------------
# DATA SUMMARY
# --------------------------------------------------

st.header("📊 Restaurant Waste Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Food Prepared",
        f"{df['food_prepared_kg'].sum():.2f} kg"
    )

with col2:
    st.metric(
        "Total Food Waste",
        f"{df['food_wasted_kg'].sum():.2f} kg"
    )

with col3:
    waste_percentage = (
        df["food_wasted_kg"].sum()
        / df["food_prepared_kg"].sum()
    ) * 100

    st.metric(
        "Overall Waste %",
        f"{waste_percentage:.2f}%"
    )

with col4:
    average_waste = df["food_wasted_kg"].mean()

    st.metric(
        "Average Waste / Record",
        f"{average_waste:.2f} kg"
    )

# --------------------------------------------------
# HOTEL-WISE SUMMARY
# --------------------------------------------------

st.header("🏨 Hotel-wise Waste Analysis")

hotel_summary = (
    df.groupby("hotel_name")
    .agg(
        food_prepared_kg=("food_prepared_kg", "sum"),
        food_wasted_kg=("food_wasted_kg", "sum")
    )
    .reset_index()
)

hotel_summary["waste_percentage"] = (
    hotel_summary["food_wasted_kg"]
    / hotel_summary["food_prepared_kg"]
) * 100

st.dataframe(
    hotel_summary,
    use_container_width=True
)

# --------------------------------------------------
# WASTE TREND
# --------------------------------------------------

st.header("📈 Food Waste Trend")

trend_data = (
    df.groupby("day")["food_wasted_kg"]
    .sum()
    .reset_index()
)

trend_data = trend_data.set_index("day")

st.line_chart(
    trend_data["food_wasted_kg"]
)

# --------------------------------------------------
# HOTEL-WISE CHART
# --------------------------------------------------

st.header("🏨 Hotel-wise Food Waste")

hotel_chart = (
    df.groupby("hotel_name")["food_wasted_kg"]
    .sum()
)

st.bar_chart(hotel_chart)

# --------------------------------------------------
# DIGITAL TWIN SIMULATION
# --------------------------------------------------

st.header("🔮 Digital Twin Waste Prediction")

st.write(
    "Simulate a future restaurant scenario and estimate "
    "food waste using the trained AI model."
)

col1, col2 = st.columns(2)

with col1:

    hotel_list = sorted(
        df["hotel_name"].unique().tolist()
    )

    selected_hotel = st.selectbox(
        "Select Hotel",
        hotel_list
    )

with col2:

    selected_day = st.number_input(
        "Day",
        min_value=1,
        value=int(df["day"].max()) + 1,
        step=1
    )

food_prepared = st.number_input(
    "Food Prepared (kg)",
    min_value=0.0,
    value=50.0,
    step=1.0
)

# --------------------------------------------------
# HOTEL CODE
# --------------------------------------------------

# Same encoding approach used during model training
hotel_categories = (
    df["hotel_name"]
    .astype("category")
)

hotel_mapping = {
    hotel: code
    for code, hotel in enumerate(
        hotel_categories.cat.categories
    )
}

hotel_code = hotel_mapping[selected_hotel]

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Food Waste"):

    input_data = pd.DataFrame({
        "day": [selected_day],
        "hotel_code": [hotel_code],
        "food_prepared_kg": [food_prepared]
    })

    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    waste_rate = (
        prediction / food_prepared * 100
        if food_prepared > 0
        else 0
    )

    st.success(
        f"Predicted Food Waste: {prediction:.2f} kg"
    )

    # --------------------------------------------------
    # WASTE LEVEL
    # --------------------------------------------------

    if waste_rate < 10:
        level = "Low 🟢"
    elif waste_rate < 15:
        level = "Moderate 🟡"
    else:
        level = "High 🔴"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Waste",
            f"{prediction:.2f} kg"
        )

    with col2:
        st.metric(
            "Waste Percentage",
            f"{waste_rate:.2f}%"
        )

    with col3:
        st.metric(
            "Waste Level",
            level
        )

    # --------------------------------------------------
    # DIGITAL TWIN SCENARIO
    # --------------------------------------------------

    st.subheader("🧠 Digital Twin Scenario")

    st.write(
        f"For **{selected_hotel}** on simulated Day "
        f"**{selected_day}**, preparing "
        f"**{food_prepared:.2f} kg** of food is predicted "
        f"to generate approximately "
        f"**{prediction:.2f} kg** of waste."
    )

    # --------------------------------------------------
    # PREPARATION OPTIMIZATION
    # --------------------------------------------------

    st.subheader("⚙️ Preparation Optimization")

    recommended_preparation = (
        food_prepared - prediction * 0.5
    )

    recommended_preparation = max(
        recommended_preparation,
        1.0
    )

    optimized_input = pd.DataFrame({
        "day": [selected_day],
        "hotel_code": [hotel_code],
        "food_prepared_kg": [recommended_preparation]
    })

    optimized_waste = model.predict(
        optimized_input
    )[0]

    optimized_waste = max(
        0,
        optimized_waste
    )

    waste_reduction = max(
        0,
        prediction - optimized_waste
    )

    reduction_percentage = (
        waste_reduction / prediction * 100
        if prediction > 0
        else 0
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Current Preparation",
            f"{food_prepared:.2f} kg"
        )

    with col2:
        st.metric(
            "Suggested Preparation",
            f"{recommended_preparation:.2f} kg"
        )

    with col3:
        st.metric(
            "Estimated Reduction",
            f"{waste_reduction:.2f} kg"
        )

    st.info(
        f"Estimated waste reduction: "
        f"**{reduction_percentage:.2f}%** "
        f"under this simulated scenario."
    )

# --------------------------------------------------
# WASTE UTILIZATION
# --------------------------------------------------

st.header("♻️ Waste Utilization Recommendation")

waste_type = st.selectbox(
    "Select Waste Type",
    [
        "Vegetable Waste",
        "Fruit Waste",
        "Cooked Food Waste",
        "Rice Waste",
        "Other Organic Waste"
    ]
)

quantity = st.number_input(
    "Waste Quantity (kg)",
    min_value=0.0,
    value=1.0,
    step=0.5
)

if st.button("💡 Get Utilization Recommendation"):

    if waste_type == "Vegetable Waste":

        recommendation = (
            "Suitable for composting or other approved "
            "organic waste processing."
        )

    elif waste_type == "Fruit Waste":

        recommendation = (
            "Can be directed toward composting or "
            "organic waste processing."
        )

    elif waste_type == "Cooked Food Waste":

        recommendation = (
            "Prioritize safe food recovery where legally "
            "and hygienically appropriate; otherwise use "
            "approved organic waste processing."
        )

    elif waste_type == "Rice Waste":

        recommendation = (
            "Can be considered for composting or other "
            "approved organic waste processing."
        )

    else:

        recommendation = (
            "Separate the organic fraction and send it "
            "for appropriate waste processing."
        )

    st.success(
        f"{quantity:.2f} kg of {waste_type}: "
        f"{recommendation}"
    )

# --------------------------------------------------
# WASTE-TO-BIOGAS SIMULATION
# --------------------------------------------------

st.header("🔥 Waste-to-Biogas Simulation")

st.write(
    "Estimate the potential biogas and methane energy "
    "from suitable organic food waste."
)

biogas_waste = st.number_input(
    "Food Waste Available for Biogas (kg)",
    min_value=0.0,
    value=5.0,
    step=1.0
)

suitable_percentage = st.slider(
    "Percentage Suitable for Anaerobic Digestion (%)",
    min_value=0,
    max_value=100,
    value=80
)

# Simulation assumptions
min_biogas_yield = 0.10
max_biogas_yield = 0.20

methane_percentage = 60

methane_energy = 9.97

# Suitable waste
suitable_waste = (
    biogas_waste
    * suitable_percentage
    / 100
)

# Biogas
min_biogas = (
    suitable_waste
    * min_biogas_yield
)

max_biogas = (
    suitable_waste
    * max_biogas_yield
)

# Methane
min_methane = (
    min_biogas
    * methane_percentage
    / 100
)

max_methane = (
    max_biogas
    * methane_percentage
    / 100
)

# Energy
min_energy = (
    min_methane
    * methane_energy
)

max_energy = (
    max_methane
    * methane_energy
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Suitable Waste",
        f"{suitable_waste:.2f} kg"
    )

with col2:
    st.metric(
        "Estimated Biogas",
        f"{min_biogas:.2f}–{max_biogas:.2f} m³"
    )

with col3:
    st.metric(
        "Estimated Methane",
        f"{min_methane:.2f}–{max_methane:.2f} m³"
    )

st.subheader("⚡ Estimated Energy Potential")

st.write(
    f"Approximately **{min_energy:.2f}–{max_energy:.2f} MJ** "
    "of methane energy could be available under the "
    "simulation assumptions."
)

st.subheader("🍳 Possible Applications")

st.write(
    "🔥 Cooking fuel — after appropriate gas treatment "
    "and safety controls"
)

st.write(
    "⚡ Electricity generation — using a suitable "
    "biogas generator"
)

st.write(
    "♨️ Heating / hot-water applications"
)

st.info(
    "These are simulation estimates, not measured biogas "
    "production. Actual output depends on waste composition, "
    "moisture, digester conditions, retention time and "
    "process efficiency."
)

# --------------------------------------------------
# COLLECTED DATA
# --------------------------------------------------

st.header("📋 Collected Restaurant Data")

st.dataframe(
    df,
    use_container_width=True
) 