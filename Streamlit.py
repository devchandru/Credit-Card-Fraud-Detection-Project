

import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('model.pkl')

st.title("💳 Credit Card Fraud Detection App")
st.subheader("Enter transaction details to check for fraud")

# Input fields
time = st.number_input('Transaction Time', min_value=0.0)
amount = st.number_input('Transaction Amount', min_value=0.0)

features = []
for i in range(1, 29):
    features.append(st.number_input(f'V{i}', value=0.0))

# Prediction
if st.button('Predict'):
    input_data = np.array([[time] + features + [amount]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error('⚠️ Fraudulent Transaction Detected!')
    else:
        st.success('✅ Legitimate Transaction')
