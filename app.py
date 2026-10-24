import streamlit as st
import joblib
import numpy as np
import pandas as pd

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
    # 1. Arrange inputs into a dictionary mapping to your exact original dataset column names
    input_dict = {
        "Product": [val1],
        "Category": [val2],
        "Region": [val3],
        "Quantity": [val4],
        "Unit_Price": [val5],
        "Discount": [val6]
    }
    
    # 2. Convert the dictionary into a pandas DataFrame
    input_df = pd.DataFrame(input_dict)
    
    # 3. Pass the DataFrame to the model instead of the numpy array
    prediction = model.predict(input_df)
    
    st.success(f"Prediction: {prediction[0]}")
