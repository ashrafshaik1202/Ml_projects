import streamlit as st
import pickle
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Uber Fare Prediction",
    page_icon="🚖",
    layout="centered"
)

# Load Model
with open("uber_fare.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("🚖 Uber Fare Prediction")

st.write("Predict the Uber fare based on the distance travelled.")

st.divider()

# User Input
distance = st.number_input(
    "Enter Distance (km)",
    min_value=0.1,
    max_value=70.0,
    value=1.0,
    step=0.1
)

# Prediction
if st.button("Predict Fare"):

    prediction = model.predict([[distance]])

    fare = prediction[0]

    st.success(f"Estimated Fare: ₹{fare:.2f}")

st.divider()

st.write("### Model Information")

st.write("- Model : Simple Linear Regression")
st.write("- Feature : Distance (km)")
st.write("- Target : Uber Fare")
st.write("- R² Score : 0.875")