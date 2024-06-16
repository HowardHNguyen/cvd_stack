import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Function to download and save the model file
def download_and_save_model(url, output_file):
    response = requests.get(url)
    response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    with open(output_file, 'wb') as f:
        f.write(response.content)

# URL for the combined model file (Update this with your GoDaddy URL)
model_url = 'https://howardnguyen.com/data/stacking_model_calibrated.pkl'

# Download and save the model file
try:
    download_and_save_model(model_url, 'stacking_model_calibrated.pkl')
    st.success("Model downloaded successfully.")
except requests.exceptions.RequestException as e:
    st.error(f"Error downloading {model_url}: {e}")

# Load the combined model
try:
    stacking_model = joblib.load('stacking_model_calibrated.pkl')
    st.success("Model loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading models: {e}")

# Load data
try:
    url = "https://raw.githubusercontent.com/HowardHNguyen/cvd/master/frmgham2.csv"
    data = pd.read_csv(url)
    data.fillna(data.mean(), inplace=True)
    st.success("Data loaded successfully.")
except Exception as e:
    st.error(f"Error loading data: {e}")

# UI setup
st.title("Cardiovascular Disease Prediction App by Howard Nguyen")

st.sidebar.header('Enter your parameters')
age = st.sidebar.slider('Enter your age:', 32, 81, 54)
totchol = st.sidebar.slider('Total Cholesterol:', 107, 696, 200)
sysbp = st.sidebar.slider('Systolic Blood Pressure:', 83, 295, 151)
diabp = st.sidebar.slider('Diastolic Blood Pressure:', 30, 150, 89)
bmi = st.sidebar.slider('BMI:', 14.43, 56.80, 26.77)
cigs_per_day = st.sidebar.slider('Cigarettes Per Day:', 0, 90, 0)
current_smoker = st.sidebar.selectbox('Current Smoker:', [0, 1])
glucose = st.sidebar.slider('Glucose:', 39, 478, 117)
diabetes = st.sidebar.selectbox('Diabetes:', [0, 1])
heartrate = st.sidebar.slider('Heart Rate:', 37, 220, 91)
bp_meds = st.sidebar.selectbox('On BP Meds:', [0, 1])
stroke = st.sidebar.selectbox('Stroke:', [0, 1])
hypertension = st.sidebar.selectbox('Hypertension:', [0, 1])

user_data = {
    'AGE': age,
    'TOTCHOL': totchol,
    'SYSBP': sysbp,
    'DIABP': diabp,
    'BMI': bmi,
    'CIGPDAY': cigs_per_day,
    'CURSMOKE': current_smoker,
    'GLUCOSE': glucose,
    'DIABETES': diabetes,
    'HEARTRTE': heartrate,
    'BPMEDS': bp_meds,
    'PREVAP': stroke,
    'HYPERTEN': hypertension
}

features = pd.DataFrame(user_data, index=[0])
st.write(features)

if st.button('Predict'):
    try:
        prediction = stacking_model.predict(features)
        prediction_proba = stacking_model.predict_proba(features)
        st.subheader('Predictions')
        st.write(f'Stacking Model Prediction: CVD with probability {prediction_proba[0][1]:.2f}')
        
        st.subheader('Prediction Probability Distribution')
        fig, ax = plt.subplots()
        ax.bar(['Stacking Model'], prediction_proba[0])
        st.pyplot(fig)
        
        st.subheader('Feature Importances (Stacking Model)')
        importances = stacking_model.named_estimators_['randomforestclassifier'].feature_importances_
        indices = np.argsort(importances)
        plt.figure()
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [features.columns[i] for i in indices])
        st.pyplot(plt)
        
        st.subheader('Model Performance')
        fig, ax = plt.subplots()
        # Replace with actual ROC plotting logic
        ax.plot([0, 1], [0, 1], 'k--')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error making predictions: {e}")
