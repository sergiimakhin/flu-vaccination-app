import streamlit as st
import pickle
import numpy as np

#############################################
# FUNCTIONS
#############################################
def encode5(x):
    mapping = {
        "Not at all effective": 1, "Not very effective": 2, "Don't know": 3,
        "Somewhat effective": 4, "Very effective": 5,

        "Very low": 1, "Somewhat low": 2, "Don't know": 3,
        "Somewhat high": 4, "Very high": 5,

        "Not at all worried": 1, "Not very worried": 2, "Don't know": 3,
        "Somewhat worried": 4, "Very worried": 5
    }
    return mapping.get(x, 3)

#############################################
# LOAD MODELS
#############################################
with open("h1n1_model.pkl", "rb") as f:
    h1n1_model = pickle.load(f)

with open("seasonal_model.pkl", "rb") as f:
    seasonal_model = pickle.load(f)


#############################################
# DEFAULT VALUES
#############################################
defaults = {
    "sex_h1": "Male",
    "age_h1": "18-34",
    "edu_h1": "Unknown",
    "ins_h1": "Yes",
    "hw_h1": "No",
    "dr_h1": "No",
    "eff_h1": "Don't know",
    "risk_h1": "Don't know",
    "sick_h1": "Don't know",

    "sex_seas": "Male",
    "age_seas": "18-34",
    "edu_seas": "Unknown",
    "ins_seas": "Yes",
    "hw_seas": "No",
    "dr_seas": "No",
    "eff_seas": "Don't know",
    "risk_seas": "Don't know",
    "sick_seas": "Don't know",
}

#############################################
# INITIALIZE SESSION VALUES
#############################################
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

#############################################
# RESET
#############################################
if st.button("RESET", key="reset"):
    for k, v in defaults.items():
        st.session_state[k] = v


#############################################
# UI TITLE
#############################################
st.write("## Vaccine Uptake Prediction")

#############################################
# 3-COLUMN LAYOUT
#############################################
left, spacer, right = st.columns([1.5, 0.3, 1.5])


#############################################################
# LEFT PANEL — H1N1
#############################################################
with left:
    st.write("### H1N1")

    st.session_state.sex_h1 = st.radio("Sex", ["Male", "Female"], key="sex_h1")
    st.session_state.age_h1 = st.selectbox("Age", ["18-34","35-44","45-54","55-64","65+"], key="age_h1")
    st.session_state.edu_h1 = st.selectbox("Education", ["Unknown","< 12 Years","12 Years","Some College","College Graduate"], key="edu_h1")
    st.session_state.ins_h1 = st.radio("Health insurance", ["Yes", "No"], key="ins_h1")
    st.session_state.hw_h1 = st.radio("Health worker", ["Yes", "No"], key="hw_h1")
    st.session_state.dr_h1 = st.radio("Doctor recommendation", ["Yes", "No"], key="dr_h1")
    st.session_state.eff_h1 = st.selectbox("Perceived effectiveness", ["Not at all effective","Not very effective","Don't know","Somewhat effective","Very effective"], key="eff_h1")
    st.session_state.risk_h1 = st.selectbox("Risk if unvaccinated", ["Very low","Somewhat low","Don't know","Somewhat high","Very high"], key="risk_h1")
    st.session_state.sick_h1 = st.selectbox("Worried about side effects", ["Not at all worried","Not very worried","Don't know","Somewhat worried","Very worried"], key="sick_h1")

    if st.button("Predict H1N1"):
        doctor_recc_h1n1 = 1.0 if st.session_state.dr_h1 == "Yes" else 0.0
        health_insurance_h1 = 1.0 if st.session_state.ins_h1 == "Yes" else 0.0
        health_worker_h1 = 1.0 if st.session_state.hw_h1 == "Yes" else 0.0
        sex_Male_h1 = True if st.session_state.sex_h1 == "Male" else False
        age_encoded_h1 = ["18-34","35-44","45-54","55-64","65+"].index(st.session_state.age_h1) + 1
        education_encoded_h1 = ["Unknown","< 12 Years","12 Years","Some College","College Graduate"].index(st.session_state.edu_h1)

        h1n1_input = np.array([[doctor_recc_h1n1,
                                health_insurance_h1,
                                encode5(st.session_state.eff_h1),
                                encode5(st.session_state.risk_h1),
                                encode5(st.session_state.sick_h1),
                                age_encoded_h1,
                                sex_Male_h1,
                                health_worker_h1,
                                education_encoded_h1]])

        prob = h1n1_model.predict_proba(h1n1_input)[0][1]
        st.write(f"**Probability: {prob:.2%}**")


#############################################################
# SPACER
#############################################################
with spacer:
    st.write(" ")


#############################################################
# RIGHT PANEL — SEASONAL
#############################################################
with right:
    st.write("### Seasonal")

    st.session_state.sex_seas = st.radio("Sex", ["Male", "Female"], key="sex_seas")  # ignored
    st.session_state.age_seas = st.selectbox("Age", ["18-34","35-44","45-54","55-64","65+"], key="age_seas")
    st.session_state.edu_seas = st.selectbox("Education", ["Unknown","< 12 Years","12 Years","Some College","College Graduate"], key="edu_seas")
    st.session_state.ins_seas = st.radio("Health insurance", ["Yes", "No"], key="ins_seas")
    st.session_state.hw_seas = st.radio("Health worker", ["Yes", "No"], key="hw_seas")
    st.session_state.dr_seas = st.radio("Doctor recommendation", ["Yes", "No"], key="dr_seas")
    st.session_state.eff_seas = st.selectbox("Perceived effectiveness", ["Not at all effective","Not very effective","Don't know","Somewhat effective","Very effective"], key="eff_seas")
    st.session_state.risk_seas = st.selectbox("Risk if unvaccinated", ["Very low","Somewhat low","Don't know","Somewhat high","Very high"], key="risk_seas")
    st.session_state.sick_seas = st.selectbox("Worried about side effects", ["Not at all worried","Not very worried","Don't know","Somewhat worried","Very worried"], key="sick_seas")

    if st.button("Predict Seasonal"):
        doctor_recc_seas = 1.0 if st.session_state.dr_seas == "Yes" else 0.0
        health_insurance_seas = 1.0 if st.session_state.ins_seas == "Yes" else 0.0
        health_worker_seas = 1.0 if st.session_state.hw_seas == "Yes" else 0.0
        age_encoded_seas = ["18-34","35-44","45-54","55-64","65+"].index(st.session_state.age_seas) + 1
        education_encoded_seas = ["Unknown","< 12 Years","12 Years","Some College","College Graduate"].index(st.session_state.edu_seas)

        seasonal_input = np.array([[doctor_recc_seas,
                                    age_encoded_seas,
                                    health_insurance_seas,
                                    encode5(st.session_state.eff_seas),
                                    encode5(st.session_state.risk_seas),
                                    encode5(st.session_state.sick_seas),
                                    health_worker_seas,
                                    education_encoded_seas]])

        prob = seasonal_model.predict_proba(seasonal_input)[0][1]
        st.write(f"**Probability: {prob:.2%}**")
