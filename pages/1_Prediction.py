import streamlit as st
import pickle
import numpy as np

# Load models
with open("h1n1_model.pkl", "rb") as f:
    h1n1_model = pickle.load(f)

with open("seasonal_model.pkl", "rb") as f:
    seasonal_model = pickle.load(f)

st.title("🧪 Vaccine Uptake Prediction")

#############################################
# FUNCTIONS
#############################################
encode5 = lambda x: {
    "Not at all effective":1, "Not very effective":2, "Don't know":3,
    "Somewhat effective":4, "Very effective":5,

    "Very low":1, "Somewhat low":2, "Don't know":3,
    "Somewhat high":4, "Very high":5,

    "Not at all worried":1, "Not very worried":2, "Don't know":3,
    "Somewhat worried":4, "Very worried":5
}.get(x, 3)  # fallback = Don't know

#############################################
# SHARED INPUTS ACROSS MODELS
#############################################
st.subheader("🧍 Respondent Information")

age_group = st.selectbox("Age group", [
    "18-34",
    "35-44",
    "45-54",
    "55-64",
    "65+"
], key="age")

sex = st.radio("Sex", ["Male", "Female"], key="sex")

health_insurance = st.radio("Health insurance", ["Yes", "No"], key="ins")

health_worker = st.radio("Healthcare worker", ["Yes", "No"], key="hw")

education = st.selectbox("Education", [
    "Unknown",
    "< 12 Years",
    "12 Years",
    "Some College",
    "College Graduate"
], key="edu")

# Encode shared data
age_encoded = ["18-34", "35-44", "45-54", "55-64", "65+"].index(age_group) + 1
education_encoded = ["Unknown","< 12 Years","12 Years","Some College","College Graduate"].index(education)
sex_Male = True if sex == "Male" else False
health_insurance = 1.0 if health_insurance == "Yes" else 0.0
health_worker = 1.0 if health_worker == "Yes" else 0.0

#############################################
# H1N1 MODEL
#############################################
st.header("🧪 H1N1 Vaccine Prediction")

doctor_recc_h1n1 = st.radio("Doctor recommendation", ["Yes", "No"], key="dr_h1")
doctor_recc_h1n1 = 1.0 if doctor_recc_h1n1 == "Yes" else 0.0

op_h1n1_effective = st.selectbox(
    "Opinion: H1N1 vaccine effectiveness",
    ["Not at all effective", "Not very effective", "Don't know",
     "Somewhat effective", "Very effective"],
    key="eff_h1"
)
op_h1n1_risk = st.selectbox(
    "Opinion: risk of flu if unvaccinated (H1N1)",
    ["Very low", "Somewhat low", "Don't know",
     "Somewhat high", "Very high"],
    key="risk_h1"
)
op_h1n1_sick = st.selectbox(
    "Worried about getting sick *from* the H1N1 vaccine",
    ["Not at all worried", "Not very worried", "Don't know",
     "Somewhat worried", "Very worried"],
    key="sick_h1"
)

h1n1_input = np.array([[doctor_recc_h1n1,
                        health_insurance,
                        encode5(op_h1n1_effective),
                        encode5(op_h1n1_risk),
                        encode5(op_h1n1_sick),
                        age_encoded,
                        sex_Male,
                        health_worker,
                        education_encoded]])

if st.button("Predict H1N1 Vaccine Uptake", key="pred_h1"):
    prob = h1n1_model.predict_proba(h1n1_input)[0][1]
    st.success(f"Probability of receiving H1N1 vaccine: {prob:.2%}")
    st.progress(float(prob))

#############################################
# SEASONAL MODEL
#############################################
st.header("💉 Seasonal Flu Vaccine Prediction")

doctor_recc_seasonal = st.radio("Doctor recommendation", ["Yes", "No"], key="dr_seas")
doctor_recc_seasonal = 1.0 if doctor_recc_seasonal == "Yes" else 0.0

op_seas_effective = st.selectbox(
    "Opinion: Seasonal vaccine effectiveness",
    ["Not at all effective", "Not very effective", "Don't know",
     "Somewhat effective", "Very effective"],
    key="eff_seas"
)
op_seas_risk = st.selectbox(
    "Opinion: risk of flu if unvaccinated (Seasonal)",
    ["Very low", "Somewhat low", "Don't know",
     "Somewhat high", "Very high"],
    key="risk_seas"
)
op_seas_sick = st.selectbox(
    "Worried about getting sick *from* the Seasonal vaccine",
    ["Not at all worried", "Not very worried", "Don't know",
     "Somewhat worried", "Very worried"],
    key="sick_seas"
)

seasonal_input = np.array([[doctor_recc_seasonal,
                            age_encoded,
                            health_insurance,
                            encode5(op_seas_effective),
                            encode5(op_seas_risk),
                            encode5(op_seas_sick),
                            health_worker,
                            education_encoded]])

if st.button("Predict Seasonal Vaccine Uptake", key="pred_seas"):
    prob = seasonal_model.predict_proba(seasonal_input)[0][1]
    st.success(f"Probability of receiving Seasonal flu vaccine: {prob:.2%}")
    st.progress(float(prob))
