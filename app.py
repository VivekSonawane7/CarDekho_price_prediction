# ============================================================
# Car Price Prediction Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------
# Page Configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

st.title("Used Car Price Prediction App")
st.markdown("Predict resale value using Machine Learning Model")

# ------------------------------------------------------------
# Load Model & Dataset
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("car_price_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("cardekho_imputated.csv")

model = load_model()
df = load_data()

# Clean dataset same way as training
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

df = df.drop_duplicates().reset_index(drop=True)

# ------------------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------------------

st.sidebar.header("Enter Car Details")

car_name = st.sidebar.selectbox(
    "Car Name",
    sorted(df["car_name"].unique())
)

brand = st.sidebar.selectbox(
    "Brand",
    sorted(df["brand"].unique())
)

model_name = st.sidebar.selectbox(
    "Model",
    sorted(df["model"].unique())
)

vehicle_age = st.sidebar.slider("Vehicle Age (Years)", 0, 30, 5)

km_driven = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    value=50000
)

mileage = st.sidebar.number_input(
    "Mileage (km/l)",
    min_value=0.0,
    value=18.0
)

engine = st.sidebar.number_input(
    "Engine (CC)",
    min_value=500,
    value=1200
)

max_power = st.sidebar.number_input(
    "Max Power (bhp)",
    min_value=30.0,
    value=80.0
)

seats = st.sidebar.selectbox(
    "Number of Seats",
    sorted(df["seats"].unique())
)

seller_type = st.sidebar.selectbox(
    "Seller Type",
    sorted(df["seller_type"].unique())
)

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    sorted(df["fuel_type"].unique())
)

transmission_type = st.sidebar.selectbox(
    "Transmission Type",
    sorted(df["transmission_type"].unique())
)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

if st.button("Predict Price"):

    input_df = pd.DataFrame([{
        "car_name": car_name,
        "brand": brand,
        "model": model_name,
        "vehicle_age": vehicle_age,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
        "seller_type": seller_type,
        "fuel_type": fuel_type,
        "transmission_type": transmission_type
    }])

    prediction = model.predict(input_df)[0]

    st.subheader("💰 Estimated Selling Price")
    st.success(f"₹ {prediction:,.0f}")

    st.info("""
    ⚠️ Prediction is based on historical CarDekho listings and may vary
    depending on market demand and car condition.
    """)

# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.markdown("---")
st.markdown("Built with Vivek using Scikit-Learn & Streamlit")