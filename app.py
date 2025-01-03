import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the pre-trained models
ridge_best_model = joblib.load('ridge_best_model.pkl')
ridge_best_ped_model = joblib.load('ridge_best_ped_model.pkl')

# Function to make predictions for the Revenue model
def predict_revenue(input_data):
    # Scale the input data
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform([input_data])
    
    # Predict using the Ridge model
    prediction = ridge_best_model.predict(input_scaled)
    return prediction[0]

# Function to make predictions for the PED model
def predict_ped(input_data):
    # Scale the input data
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform([input_data])
    
    # Predict using the PED model
    prediction = ridge_best_ped_model.predict(input_scaled)
    return prediction[0]

# Streamlit app layout
st.title("Revenue and PED Prediction App")

st.sidebar.header("Input Features")

# Create sliders or input fields for users to input the data
old_price = st.sidebar.number_input("Old Price", min_value=0, max_value=1000, value=100)
new_price = st.sidebar.number_input("New Price", min_value=0, max_value=1000, value=90)
old_quantity = st.sidebar.number_input("Old Quantity", min_value=0, max_value=1000, value=300)
new_quantity = st.sidebar.number_input("New Quantity", min_value=0, max_value=1000, value=350)

# Calculate the feature values based on the input
price_change = ((new_price - old_price) / old_price) * 100
quantity_change = ((new_quantity - old_quantity) / old_quantity) * 100
price_to_quantity_ratio = old_price / old_quantity
price_quantity_interaction = old_price * old_quantity

# Input data for prediction
input_data = [price_change, quantity_change, price_to_quantity_ratio, price_quantity_interaction]

# Prediction buttons for Revenue and PED
if st.sidebar.button('Predict Revenue Change'):
    revenue_prediction = predict_revenue(input_data)
    st.subheader("Predicted Revenue Change")
    st.write(f"The predicted revenue change is: {revenue_prediction:.2f}")

if st.sidebar.button('Predict Price Elasticity of Demand (PED)'):
    ped_prediction = predict_ped(input_data)
    st.subheader("Predicted Price Elasticity of Demand (PED)")
    st.write(f"The predicted PED is: {ped_prediction:.2f}")

