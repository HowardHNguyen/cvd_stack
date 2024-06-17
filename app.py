import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

# URL for the combined model file
model_url = 'https://howardnguyen.com/data/stacking_model_calibrated.pkl'

@st.cache_resource
def load_model(url):
    response = requests.get(url)
    response.raise_for_status()
    model_file = BytesIO(response.content)
    model = joblib.load(model_file)
    return model

stacking_model = load_model(model_url)

st.title("Cardiovascular Disease Prediction App by Howard Nguyen")
st.write("Enter your parameters and click Predict to get the results.")

# Create two columns for the layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Enter your age:", 32, 81, 54)
    totchol = st.slider("Total Cholesterol:", 107, 696, 200)
    sysbp = st.slider("Systolic Blood Pressure:", 83, 295, 151)
    diabp = st.slider("Diastolic Blood Pressure:", 30, 150, 89)
    bmi = st.slider("BMI:", 14.43, 56.80, 26.77)
    cigpday = st.slider("Cigarettes Per Day:", 0, 90, 20)
    cursmoke = st.selectbox("Current Smoker:", [0, 1])
    bpmeds = st.selectbox("On BP Meds:", [0, 1])
    hyperten = st.selectbox("Hypertension:", [0, 1])

with col2:
    glucose = st.slider("Glucose:", 39, 478, 117)
    diabetes = st.selectbox("Diabetes:", [0, 1])
    heartrate = st.slider("Heart Rate:", 37, 220, 91)
    prevap = st.selectbox("Prevalent Ap:", [0, 1])
    stroke = st.selectbox("Stroke:", [0, 1])

# Constructing DataFrame for prediction
input_data = pd.DataFrame({
    'AGE': [age],
    'TOTCHOL': [totchol],
    'SYSBP': [sysbp],
    'DIABP': [diabp],
    'BMI': [bmi],
    'CIGPDAY': [cigpday],
    'CURSMOKE': [cursmoke],
    'GLUCOSE': [glucose],
    'DIABETES': [diabetes],
    'HEARTRTE': [heartrate],
    'PREVAP': [prevap],
    'STROKE': [stroke],
    'BPMEDS': [bpmeds],
    'HYPERTEN': [hyperten]
})

st.write("### Input Parameters")
st.write(input_data)

# Prediction
if st.button("Predict"):
    try:
        predictions = stacking_model.predict_proba(input_data)
        rf_pred = predictions[:, 1]
        
        st.write(f"Stacking Model Prediction: CVD with probability {rf_pred[0]:.2f}")
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")
