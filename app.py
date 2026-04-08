import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.title("🏠 House Price Prediction")

# Inputs
area = st.number_input("Area")
bedrooms = st.number_input("Bedrooms")
bathrooms = st.number_input("Bathrooms")
stories = st.number_input("Stories")
parking = st.number_input("Parking")

mainroad = st.selectbox("Main Road", ["Yes", "No"])
guestroom = st.selectbox("Guest Room", ["Yes", "No"])
basement = st.selectbox("Basement", ["Yes", "No"])
hotwaterheating = st.selectbox("Hot Water Heating", ["Yes", "No"])
airconditioning = st.selectbox("Air Conditioning", ["Yes", "No"])
prefarea = st.selectbox("Preferred Area", ["Yes", "No"])
furnishingstatus = st.selectbox(
    "Furnishing Status",
    ["furnished", "semi-furnished", "unfurnished"]
)

# Create dataframe
input_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'parking': [parking],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})

# Convert to dummy variables
input_data = pd.get_dummies(input_data)

# Align with model
# Match training columns manually
expected_columns = [
    'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
    'mainroad_yes', 'guestroom_yes', 'basement_yes',
    'hotwaterheating_yes', 'airconditioning_yes',
    'prefarea_yes',
    'furnishingstatus_semi-furnished',
    'furnishingstatus_unfurnished'
]

input_data = input_data.reindex(columns=expected_columns, fill_value=0)
# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Price: ₹ {int(prediction[0])}")