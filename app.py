import streamlit as st
import joblib
import numpy as np

# Load the pre-trained models
ped_model = joblib.load('ridge_best_ped_model.pkl')
revenue_model = joblib.load('ridge_best_model.pkl')

# Streamlit Interface for the client
def predict_ped_and_revenue(old_price, new_price, old_qty, new_qty):
    # Prepare the input data for prediction
    price_change = ((new_price - old_price) / old_price) * 100
    quantity_change = ((new_qty - old_qty) / old_qty) * 100
    price_to_quantity_ratio = old_price / old_qty
    price_quantity_interaction = old_price * old_qty
    input_data = np.array([[price_change, quantity_change, price_to_quantity_ratio, price_quantity_interaction]])

    # Predict PED and Revenue Change
    predicted_ped = ped_model.predict(input_data)[0]
    predicted_revenue_change = revenue_model.predict(input_data)[0]

    # Display results
    st.write(f"Predicted PED: {predicted_ped:.2f}")
    st.write(f"Predicted Revenue Change: {predicted_revenue_change:.2f}")

# Streamlit app
st.title("Price Elasticity of Demand (PED) and Revenue Forecast")
st.write("Enter old price, new price, old quantity, and new quantity to calculate PED and revenue change.")

old_price = st.number_input("Enter Old Price", min_value=1)
new_price = st.number_input("Enter New Price", min_value=1)
old_qty = st.number_input("Enter Old Quantity", min_value=1)
new_qty = st.number_input("Enter New Quantity", min_value=1)

if st.button('Calculate'):
    predict_ped_and_revenue(old_price, new_price, old_qty, new_qty)
