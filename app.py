import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🏠 House Price Prediction")

area = st.number_input("Area (sq ft)", 500, 10000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
stories = st.number_input("Stories", 1, 5)
parking = st.number_input("Parking Spaces", 0, 5)

if st.button("Predict Price"):
    data = np.array([[area, bedrooms, bathrooms, stories, parking]])
    data_scaled = scaler.transform(data)
    price = model.predict(data_scaled)
    st.success(f"Estimated Price: ₹ {price[0]:,.2f}")