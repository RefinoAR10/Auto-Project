# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,root_mean_squared_error
import joblib
import os

# -----------------------------
# 1. Load CSV (no header)
# -----------------------------
data = pd.read_csv("auto_imports.csv", header=None)

# Assign proper column names
columns = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
           "num_doors", "body_style", "drive_wheels", "engine_location",
           "wheel_base", "length", "width", "height", "curb_weight",
           "engine_type", "num_cylinders", "engine_size", "fuel_system",
           "bore", "stroke", "compression_ratio", "horsepower", "peak_rpm",
           "city_mpg", "highway_mpg", "price"]

data.columns = columns
print("Dataset loaded successfully!")

# -----------------------------
# 2. Clean data
# -----------------------------
# Replace '?' with NaN
data.replace('?', pd.NA, inplace=True)

# Drop rows where target 'price' is missing
data.dropna(subset=["price"], inplace=True)

# Convert numeric columns to float
numeric_cols = ["normalized_losses", "wheel_base", "length", "width", "height",
                "curb_weight", "engine_size", "bore", "stroke", "compression_ratio",
                "horsepower", "peak_rpm", "city_mpg", "highway_mpg", "price"]

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Fill remaining missing numeric values with median
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Optional: fill missing categorical values with mode
categorical_cols = ["make", "fuel_type", "aspiration", "num_doors", "body_style",
                    "drive_wheels", "engine_location", "engine_type", "num_cylinders", "fuel_system"]

for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

# -----------------------------
# 3. Prepare features and target
# -----------------------------
x = data.drop("price", axis=1)
y = data["price"]

# Convert categorical columns using one-hot encoding
x = pd.get_dummies(x, drop_first=True)

# Save column names for later use in prediction
joblib.dump(x.columns.tolist(), "columns.pkl")

# -----------------------------
# 4. Train/test split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# -----------------------------
# 5. Train Random Forest model
# -----------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# -----------------------------
# 6. Evaluate model
# -----------------------------
# Make predictions
y_pred = model.predict(x_test)

# Compute MSE
mse = mean_squared_error(y_test, y_pred)

# Print evaluation metrics
from sklearn.metrics import r2_score
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")


# -----------------------------
# 7. Save trained model
# -----------------------------
joblib.dump(model, "model.pkl")
print("Model and columns saved successfully!")