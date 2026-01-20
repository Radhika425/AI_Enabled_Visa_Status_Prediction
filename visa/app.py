import streamlit as st
import pandas as pd
import joblib

# Load models
lower_model = joblib.load("lower_quantile_model.pkl")
median_model = joblib.load("median_model.pkl")
upper_model = joblib.load("upper_quantile_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Visa Processing Time Estimator", layout="centered")

st.title("🛂 AI Visa Processing Time Estimator")
st.write("Enter application details to estimate processing time")

# -----------------------------
# USER INPUTS
# -----------------------------
continent = st.selectbox("Continent", [0, 1, 2, 3, 4])
education_level = st.selectbox("Education Level", [0, 1, 2, 3, 4])
has_job_e = st.radio("Has Job Experience?", [0, 1])
requires_j = st.radio("Job Requires Experience?", [0, 1])
full_time = st.radio("Full-Time Position?", [0, 1])
no_of_em = st.number_input("Number of Employees", min_value=1)
company_age = st.number_input("Company Age (years)", min_value=0)
prevailing_wage = st.number_input("Prevailing Wage (Yearly)", min_value=0)
region = st.selectbox("Employment Region", [0, 1, 2, 3])
unit_of_w = st.selectbox("Wage Unit", [0, 1])

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("Estimate Processing Time"):

    input_df = pd.DataFrame([{
        "continent": continent,
        "education_level": education_level,
        "has_job_e": has_job_e,
        "requires_j": requires_j,
        "full_time_": full_time,
        "no_of_em": no_of_em,
        "company_age": company_age,
        "prevailing_": prevailing_wage,
        "region_of_": region,
        "unit_of_w": unit_of_w
    }])

    # Scale input
    input_scaled = scaler.transform(input_df)

    lower = lower_model.predict(input_scaled)[0]
    median = median_model.predict(input_scaled)[0]
    upper = upper_model.predict(input_scaled)[0]

    st.success(f"⏱ Estimated Processing Time: **{int(median)} days**")
    st.info(f"📊 Expected Range: **{int(lower)} – {int(upper)} days**")
    st.caption("Confidence Level: 80%")
