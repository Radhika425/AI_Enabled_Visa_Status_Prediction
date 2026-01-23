import streamlit as st
import pandas as pd
import joblib
import os

# ---------------------------
# Load model bundle (FIXED PATH)
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoders = bundle["encoders"]

st.title("Visa Case Status Predictor (Regression Based)")

# ---------------------------
# User Inputs
# ---------------------------
education = st.selectbox(
    "Education of Employee",
    encoders["education_of_employee"].classes_
)

job_exp = st.selectbox(
    "Has Job Experience",
    encoders["has_job_experience"].classes_
)

training = st.selectbox(
    "Requires Job Training",
    encoders["requires_job_training"].classes_
)

wage = st.number_input("Prevailing Wage", min_value=0.0)

unit = st.selectbox(
    "Unit of Wage",
    encoders["unit_of_wage"].classes_
)

full_time = st.selectbox(
    "Full Time Position",
    encoders["full_time_position"].classes_
)

# ---------------------------
# Predict
# ---------------------------
if st.button("Predict Case Status"):
    input_df = pd.DataFrame([{
        "education_of_employee": encoders["education_of_employee"].transform([education])[0],
        "has_job_experience": encoders["has_job_experience"].transform([job_exp])[0],
        "requires_job_training": encoders["requires_job_training"].transform([training])[0],
        "prevailing_wage": wage,
        "unit_of_wage": encoders["unit_of_wage"].transform([unit])[0],
        "full_time_position": encoders["full_time_position"].transform([full_time])[0]
    }])

    prediction = model.predict(input_df)[0]

    result = "Certified" if prediction >= 0.5 else "Denied"

    st.success(f"Predicted Case Status: **{result}**")
