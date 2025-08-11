import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import streamlit as st

# Load your trained model
model = joblib.load(open('final_model.pkl', 'rb'))

st.title("💓 Heart Failure Prediction App")

st.markdown("Fill out all the fields below to get a prediction.")

# --- Form Inputs ---
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
resting_bp = st.number_input("Resting Blood Pressure", min_value=0, value=120)
cholesterol = st.number_input("Cholesterol", min_value=0, value=200)
fasting_bs = st.selectbox("Fasting Blood Sugar", [("> 120 mg/dl", 1), ("≤ 120 mg/dl", 0)], format_func=lambda x: x[0])[1]
max_hr = st.number_input("Max Heart Rate", min_value=0, value=150)
exercise_angina = st.selectbox("Exercise-Induced Angina", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
oldpeak = st.number_input("Oldpeak (ST Depression)", step=0.1, value=1.0)
st_slope = st.selectbox("ST Slope", [("Up", 2), ("Flat", 1), ("Down", 0)], format_func=lambda x: x[0])[1]

# Hidden cholesterol_is_zero field
cholesterol_is_zero = 0

# Chest Pain Types
chest_asy = st.selectbox("Chest Pain (ASY)", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
chest_ata = st.selectbox("Chest Pain (ATA)", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
chest_nap = st.selectbox("Chest Pain (NAP)", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
chest_ta = st.selectbox("Chest Pain (TA)", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]

# Resting ECG
ecg_lvh = st.selectbox("Resting ECG (LVH)", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
ecg_normal = st.selectbox("Resting ECG (Normal)", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
ecg_st = st.selectbox("Resting ECG (ST)", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]

# --- Predict Button ---
if st.button("Predict"):
    # Order must match training dataset
    features = np.array([
        age, sex, resting_bp, cholesterol, fasting_bs, max_hr, exercise_angina,
        oldpeak, st_slope, cholesterol_is_zero,
        chest_asy, chest_ata, chest_nap, chest_ta,
        ecg_lvh, ecg_normal, ecg_st
    ]).reshape(1, -1)

    # Predict probability
    y_proba = model.predict_proba(features)[:, 1]

    # Apply threshold
    y_pred = (y_proba >= 0.3).astype(int)

    # Display result
    if y_pred[0] == 0:
        st.success(f"✅ You do NOT have Heart Failure risk (probability: {y_proba[0]:.2%})")
    else:
        st.error(f"⚠️ You HAVE Heart Failure risk (probability: {y_proba[0]:.2%})")