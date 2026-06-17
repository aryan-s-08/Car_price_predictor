import os
import subprocess
import sys

# Force the container environment to install scikit-learn right on launch
try:
    import sklearn
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn==1.6.1"])

import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Load the model and data
try:
    model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
    car = pd.read_csv('Cleaned_car.csv')
except FileNotFoundError:
    st.error("Error: Make sure 'LinearRegressionModel.pkl' and 'Cleaned_car.csv' are in the same folder.")

# 2. Configure the App Title and Description
st.title("Welcome to Car Price Predictor")
st.markdown("This app predicts the price of a car you want to sell. Try filling the details below:")

# 3. Extract uniquely sorted values for dropdown lists
companies = sorted(car['company'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

# 4. Create UI Dropdowns and Input Fields
selected_company = st.selectbox("Select the company:", companies)

# Dynamically filter models based on selected company
filtered_models = sorted(car[car['company'] == selected_company]['name'].unique())
selected_model = st.selectbox("Select the model:", filtered_models)

selected_year = st.selectbox("Select Year of Purchase:", years)
selected_fuel = st.selectbox("Select the Fuel Type:", fuel_types)
kms_driven = st.number_input("Enter the Number of Kilometres that the car has travelled:", min_value=0, value=10000, step=500)

# 5. Prediction Logic
if st.button("Predict Price", type="primary"):
    # Build input DataFrame matching the exact structure your pipeline model expects
    input_data = pd.DataFrame(
        [[selected_model, selected_company, selected_year, kms_driven, selected_fuel]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    )
    
    # Run model prediction
    prediction = model.predict(input_data)
    
    # Display Result
    st.success(f"Estimated Prediction Price: ₹{np.round(prediction[0], 2):,}")
