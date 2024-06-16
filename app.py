import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Function to combine split files
def combine_files(part1_url, part2_url, output_file):
    response1 = requests.get(part1_url)
    response2 = requests.get(part2_url)

    with open(output_file, 'wb') as f:
        f.write(response1.content)
        f.write(response2.content)

# URLs for the split parts
part1_url = 'https://raw.githubusercontent.com/HowardHNguyen/cvd_stack/master/stacking_model_calibrated_part1.pkl'
part2_url = 'https://raw.githubusercontent.com/HowardHNguyen/cvd_stack/master/stacking_model_calibrated_part2.pkl'

# Combine the split parts
combine_files(part1_url, part2_url, 'stacking_model_calibrated.pkl')

# Load the combined model
try:
    model = joblib.load('stacking_model_calibrated.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")

# Load data
try:
    url = "https://raw.githubusercontent.com/HowardHNguyen/cvd/master/frmgham2.csv"
    data = pd.read_csv(url)
    data.fillna(data.mean(), inplace=True)
except Exception as e:
    st.error(f"Error loading data: {e}")

# UI setup
st.title("Cardiovascular Disease Prediction")

st.sidebar.header('User Input Features')
age = st.sidebar.slider('Age', 0, 120, 50)
totchol = st.sidebar.slider('Total Cholesterol', 100, 600, 200)
sysbp = st.sidebar.slider('Systolic Blood Pressure', 80, 250, 120)
diabp = st.sidebar.slider('Diastolic Blood Pressure', 50, 150, 80)
bmi = st.sidebar.slider('Body Mass Index', 10.0, 60.0, 25.0)
glucose = st.sidebar.slider('Glucose', 40, 500, 90)
heartrate = st.sidebar.slider('Heart Rate', 40, 180, 70)
cig_per_day = st.sidebar.slider('Cigarettes per Day', 0, 60, 0)

# Create input data
input_data = pd.DataFrame({
    'AGE': [age],
    'TOTCHOL': [totchol],
    'SYSBP': [sysbp],
    'DIABP': [diabp],
    'BMI': [bmi],
    'GLUCOSE': [glucose],
    'HEARTRTE': [heartrate],
    'CIGPDAY': [cig_per_day]
})

st.subheader('User Input features')
st.write(input_data)

# Predict
if st.button('Predict'):
    predictions = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    st.subheader('Predictions')
    st.write(f'Prediction: {predictions[0]}')
    st.write(f'Prediction Probability: {prediction_proba[0][1]:.2f}')

    # Plot ROC Curve and Feature Importance
    st.subheader('Model Performance Metrics')

    # Example ROC curve (use actual validation data for real application)
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, 0.2, 0.8, 1], [0, 0.4, 0.6, 1], label='Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    st.pyplot(plt)

    # Example feature importance (use actual feature importance for real application)
    feature_importance = model.named_estimators_['randomforestclassifier'].feature_importances_
    sorted_idx = np.argsort(feature_importance)
    plt.figure()
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), input_data.columns[sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    st.pyplot(plt)
