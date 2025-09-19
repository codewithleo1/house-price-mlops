import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/model.pkl")

st.title("🏡 House Price Prediction App")

# User inputs
area = st.number_input("Area (sqft)", min_value=500, max_value=20000, step=100)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 10, 2)
stories = st.slider("Stories", 1, 5, 2)
mainroad = st.selectbox("Mainroad", ["yes", "no"])
guestroom = st.selectbox("Guestroom", ["yes", "no"])
basement = st.selectbox("Basement", ["yes", "no"])
hotwaterheating = st.selectbox("Hot Water Heating", ["yes", "no"])
airconditioning = st.selectbox("Air Conditioning", ["yes", "no"])
parking = st.slider("Parking", 0, 5, 1)
prefarea = st.selectbox("Preferred Area", ["yes", "no"])
furnishingstatus = st.selectbox("Furnishing Status", ["furnished", "semi-furnished", "unfurnished"])

# Encode categorical values
def encode_value(val):
    return 1 if val == "yes" else 0

input_data = pd.DataFrame([[
    area, bedrooms, bathrooms, stories,
    encode_value(mainroad),
    encode_value(guestroom),
    encode_value(basement),
    encode_value(hotwaterheating),
    encode_value(airconditioning),
    parking,
    encode_value(prefarea),
    {"furnished": 2, "semi-furnished": 1, "unfurnished": 0}[furnishingstatus]
]], columns=[
    "area","bedrooms","bathrooms","stories","mainroad","guestroom","basement",
    "hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"
])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹{prediction:,.2f}")
