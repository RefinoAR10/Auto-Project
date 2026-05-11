import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.title("Car Price Prediction App 🚗💰")
# -----------------------------
# 1. Load model and columns
# -----------------------------
model = joblib.load("model.pkl","r")
model_columns = joblib.load("columns.pkl","r")

st.header("Select Car Features")

# Create 2-column layout for better UI
col1, col2 = st.columns(2)

with col1:
    wheel_base = st.slider("Wheel Base", 80.0, 120.0, 88.6)
    length = st.slider("Length", 120.0, 200.0, 168.8)
    width = st.slider("Width", 60.0, 80.0, 64.1)
    height = st.slider("Height", 40.0, 70.0, 48.8)
    curb_weight = st.slider("Curb Weight", 1500, 4000, 2548)
    engine_size = st.slider("Engine Size", 60, 300, 130)
    bore = st.slider("Bore", 2.0, 4.0, 3.47)
    stroke = st.slider("Stroke", 2.0, 4.0, 2.68)
    compression_ratio = st.slider("Compression Ratio", 7.0, 11.0, 9.0)
    horsepower = st.slider("Horsepower", 50, 300, 111)
    peak_rpm = st.slider("Peak RPM", 4000, 7000, 5000)
    city_mpg = st.slider("City MPG", 10, 50, 21)
    highway_mpg = st.slider("Highway MPG", 15, 60, 27)


with col2:


    make = st.selectbox("Make", ["alfa-romero", "audi", "bmw", "chevrolet", "dodge", "honda",
                                 "isuzu", "jaguar", "mazda", "mercedes-benz", "mercury",
                                 "mitsubishi", "nissan", "peugot", "plymouth", "porsche",
                                 "renault", "saab", "subaru", "toyota", "volkswagen", "volvo"])

    fuel_type = st.selectbox("Fuel Type", ["gas", "diesel"])
    aspiration = st.selectbox("Aspiration", ["std", "turbo"])
    num_doors = st.selectbox("Number of Doors", ["two", "four"])
    body_style = st.selectbox("Body Style", ["convertible", "hatchback", "sedan", "wagon", "hardtop"])
    drive_wheels = st.selectbox("Drive Wheels", ["fwd", "rwd", "4wd"])
    engine_location = st.selectbox("Engine Location", ["front", "rear"])
    engine_type = st.selectbox("Engine Type", ["dohc", "ohc", "ohcf", "ohcv", "l", "rotor"])
    num_cylinders = st.selectbox("Number of Cylinders", ["two", "three", "four", "five", "six", "eight", "twelve"])
    fuel_system = st.selectbox("Fuel System", ["mpfi", "2bbl", "idi", "1bbl", "spdi", "4bbl"])

# -----------------------------
# 2. Prepare input DataFrame
# -----------------------------
input_dict = {
    "wheel_base": wheel_base,
    "length": length,
    "width": width,
    "height": height,
    "curb_weight": curb_weight,
    "engine_size": engine_size,
    "bore": bore,
    "stroke": stroke,
    "compression_ratio": compression_ratio,
    "horsepower": horsepower,
    "peak_rpm": peak_rpm,
    "city_mpg": city_mpg,
    "highway_mpg": highway_mpg,
    "make": make,
    "fuel_type": fuel_type,
    "aspiration": aspiration,
    "num_doors": num_doors,
    "body_style": body_style,
    "drive_wheels": drive_wheels,
    "engine_location": engine_location,
    "engine_type": engine_type,
    "num_cylinders": num_cylinders,
    "fuel_system": fuel_system
}

input_df = pd.DataFrame([input_dict])

# One-hot encode categorical variables
input_encoded = pd.get_dummies(input_df)

# Add missing columns (if any)
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Arrange columns correctly
input_encoded = input_encoded[model_columns]

# -----------------------------
# 3. Predict price
# -----------------------------
if st.button("Predict Price"):
    prediction = model.predict(input_encoded)[0]
    st.success(f"Estimated Car Price: ${prediction:,.2f}")



