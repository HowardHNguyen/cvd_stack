import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO
import matplotlib.pyplot as plt

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
    glucose = st.slider("Glucose:", 39, 478, 117)
    diabetes = st.selectbox("Diabetes:", [0, 1])

with col2:
    heartrate = st.slider("Heart Rate:", 37, 220, 91)
    prevap = st.selectbox("Prevalent Ap:", [0, 1])
    stroke = st.selectbox("Stroke:", [0, 1])
    bpmeds = st.selectbox("On BP Meds:", [0, 1])
    hyperten = st.selectbox("Hypertension:", [0, 1])

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
        pred_proba = predictions[:, 1][0]

        st.write("## Your CVD Probability Prediction Results")
        st.write(f"**Stacking Model Prediction:** Your CVD probability is {pred_proba:.2f}")

        # Probability Distribution
        st.write("### Prediction Probability Distribution")
        fig, ax = plt.subplots()
        ax.bar(['Stacking Model'], [pred_proba], color=['blue'])
        ax.set_ylabel('Probability')
        for i, v in enumerate([pred_proba]):
            ax.text(i, v + 0.02, f"{v:.2f}", color='black', ha='center')
        st.pyplot(fig)

        # Feature Importances
        st.write("### Feature Importances (Stacking Model)")
        try:
            importances = stacking_model.named_estimators_['randomforestclassifier'].feature_importances_
            indices = np.argsort(importances)[::-1]
            features = input_data.columns
            fig, ax = plt.subplots()
            ax.barh(range(len(indices)), importances[indices], align='center')
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels(features[indices])
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error plotting feature importances: {e}")

        # ROC Curve
        st.write("### Model Performance")
        fig, ax = plt.subplots()
        for estimator_name, est in stacking_model.named_estimators_.items():
            try:
                from sklearn.metrics import roc_curve, auc
                probas_ = est.predict_proba(input_data)[:, 1]
                fpr, tpr, _ = roc_curve([0, 1], probas_)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'{estimator_name} (AUC = {roc_auc:.2f})')
            except Exception as e:
                st.error(f"Error plotting ROC curve for {estimator_name}: {e}")
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend(loc="lower right")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error making predictions: {e}")
