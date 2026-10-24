import streamlit as st
import joblib
import numpy as np

# Load the model directly from the repository environment
model = joblib.load('salesregression.pkl')

st.title("🤖 Cloud-Hosted ML Predictor")

# Input fields
val1 = st.text_input("Feature 1", value=0.0)
val2 = st.text_input("Feature 2", value=0.0)
val3 = st.text_input("Feature 3", value=0.0)
val4 = st.number_input("Feature 4", value=0.0)
val5 = st.number_input("Feature 5", value=0.0)
val6 = st.number_input("Feature 6", value=0.0)

if st.button("Predict"):
    prediction = model.predict(np.array([text[val1, val2, val3, val4, val5, val6]]))
    st.success(f"Prediction: {prediction[0]}")
