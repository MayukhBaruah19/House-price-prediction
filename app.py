import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
with open('regression_model.pkl', "rb") as file:
    model = pickle.load(file)

# Load the encoders and scaler
with open('onehot_encoder_location.pkl', 'rb') as file:
    onehot_encoder_location = pickle.load(file)


with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


# streamlit app
st.title(' Guwahati House Price Prediction')

# User input
location = st.sidebar.selectbox(
    'location', onehot_encoder_location.categories_[0])
bhk = st.sidebar.slider('bhk', 1, 6, 3)
size = st.sidebar.slider("Size of the house (in square feet)", 0, 3000, 500)


# Prepare the input data
input_data = pd.DataFrame({
    'bhk': [bhk],
    'size': [size]
})

# One-hot encode 'location'
location_encoded = onehot_encoder_location.transform([[location]]).toarray()
location_encoded_df = pd.DataFrame(
    location_encoded, columns=onehot_encoder_location.get_feature_names_out(['location']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(
    drop=True), location_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Prediction
prediction = model.predict(input_data)


st.subheader(f"The estimated price is ₹{ prediction[0]:.2f} Lakhs")
