import streamlit as st
import joblib
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")

hours_studied = st.slider("Hours Studied", 0.0, 24.0, 2.0)
attendance_rate = st.slider("Attendance Rate (%)", 0.0, 100.0, 80.0)
mental_health = st.slider("Mental Health Score (1-10)", 1, 10, 5)
sleep_hours = st.slider("Average Sleep Hours", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])

ptj_encoded = 1 if part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):

    input_data = np.array([[hours_studied, attendance_rate, mental_health, sleep_hours, ptj_encoded]])
    prediction = model.predict(input_data)[0]

    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Exam Score: {prediction:.2f}%")



