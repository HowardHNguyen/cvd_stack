import streamlit as st
import pandas as pd
import joblib
import urllib.request

# URL to load the model
model_url = 'https://howardnguyen.com/data/stacking_model_calibrated.pkl'

# Download and load the model
try:
    with st.spinner(f'Downloading {model_url}...'):
        filename, _ = urllib.request.urlretrieve(model_url)
    with st.spinner(f'Loading model...'):
        stacking_model = joblib.load(filename)
    st.success(f'Model downloaded successfully from {model_url}')
except Exception as e:
    st.error(f"Error downloading {model_url}: {str(e)}")
    st.stop()

st.title('Cardiovascular Disease Prediction App by Howard Nguyen')
st.write('Enter your parameters and click Predict to get the results.')

col1, col2 = st.columns(2)

with col1:
    age = st.slider('Enter your age:', 32, 81, 54)
    totchol = st.slider('Total Cholesterol:', 107, 696, 200)
    sysbp = st.slider('Systolic Blood Pressure:', 83, 295, 151)
    diabp = st.slider('Diastolic Blood Pressure:', 30, 150, 89)
    bmi = st.slider('BMI:', 14.43, 56.80, 26.77)
    cigday = st.slider('Cigarettes Per Day:', 0, 90, 20)
    cursmoke = st.selectbox('Current Smoker:', [0, 1])

with col2:
    glucose = st.slider('Glucose:', 39, 478, 117)
    diabetes = st.selectbox('Diabetes:', [0, 1])
    heartrate = st.slider('Heart Rate:', 37, 220, 91)
    prevap = st.selectbox('Prevalent Ap:', [0, 1])
    stroke = st.selectbox('Stroke:', [0, 1])
    bpmeds = st.selectbox('On BP Meds:', [0, 1])
    hyperten = st.selectbox('Hypertension:', [0, 1])

input_data = pd.DataFrame({
    'AGE': [age], 'TOTCHOL': [totchol], 'SYSBP': [sysbp], 'DIABP': [diabp], 'BMI': [bmi],
    'CIGPDAY': [cigday], 'CURSMOKE': [cursmoke], 'GLUCOSE': [glucose], 'DIABETES': [diabetes],
    'HEARTRTE': [heartrate], 'PREVAP': [prevap], 'STROKE': [stroke], 'BPMEDS': [bpmeds], 'HYPERTEN': [hyperten]
})

st.subheader('Input Parameters')
st.table(input_data)

if st.button('Predict'):
    try:
        rf_proba = stacking_model.predict_proba(input_data)[:, 1]
        st.success(f'Your CVD probability prediction: {rf_proba[0]:.2f}')
    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
