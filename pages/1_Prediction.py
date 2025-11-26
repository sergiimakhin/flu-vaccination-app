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
# STYLE
#############################################
st.markdown("""
<style>
body { background-color:#FFFFFF; color:#000000; }

hr {
    border: 0;
    height: 1px;
    background: #8AC6D6;
    margin: 8px 0;
}

.feature-box {
    background-color:#58ADC5;
    padding:8px;
    border-radius:5px;
    margin-bottom:6px;
}

.question-label {
    font-weight:600;
    font-size:14px;
}

.result-box {
    background-color:#F08C66;
    color:#000000;
    padding:10px;
    border-radius:6px;
    font-size:16px;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)


#############################################
# TITLE
#############################################
st.markdown("<h1 style='text-align:center; margin-bottom:1rem;'>VACCINE UPTAKE PREDICTION</h1>", unsafe_allow_html=True)


#############################################
# 3 COLUMNS LAYOUT
#############################################
left, spacer, right = st.columns([1.5, 0.3, 1.5])

########################################################
# LEFT PANEL — H1N1 MODEL
########################################################
with left:
    st.subheader("H1N1 Model")

    # SEX
    st.markdown("<div class='feature-box'><span class='question-label'>Sex</span></div>", unsafe_allow_html=True)
    sex_h1 = st.radio("", ["Male", "Female"], key="sex_h1")

    # AGE
    st.markdown("<div class='feature-box'><span class='question-label'>Age group</span></div>", unsafe_allow_html=True)
    age_h1 = st.selectbox("", ["18-34","35-44","45-54","55-64","65+"], key="age_h1")

    # EDUCATION
    st.markdown("<div class='feature-box'><span class='question-label'>Education</span></div>", unsafe_allow_html=True)
    education_h1 = st.selectbox("", ["Unknown","< 12 Years","12 Years","Some College","College Graduate"], key="edu_h1")

    # HEALTH INSURANCE
    st.markdown("<div class='feature-box'><span class='question-label'>Health insurance</span></div>", unsafe_allow_html=True)
    health_insurance_h1 = st.radio("", ["Yes", "No"], key="ins_h1")

    # HEALTH WORKER
    st.markdown("<div class='feature-box'><span class='question-label'>Health worker</span></div>", unsafe_allow_html=True)
    health_worker_h1 = st.radio("", ["Yes", "No"], key="hw_h1")

    # DOCTOR RECOMMENDATION
    st.markdown("<div class='feature-box'><span class='question-label'>Doctor recommendation</span></div>", unsafe_allow_html=True)
    doctor_recc_h1n1 = st.radio("", ["Yes", "No"], key="dr_h1")

    # VACCINE EFFECTIVENESS
    st.markdown("<div class='feature-box'><span class='question-label'>Vaccine perceived effectiveness</span></div>", unsafe_allow_html=True)
    op_h1n1_effective = st.selectbox("", ["Not at all effective","Not very effective","Don't know","Somewhat effective","Very effective"], key="eff_h1")

    # RISK
    st.markdown("<div class='feature-box'><span class='question-label'>Risk if unvaccinated</span></div>", unsafe_allow_html=True)
    op_h1n1_risk = st.selectbox("", ["Very low","Somewhat low","Don't know","Somewhat high","Very high"], key="risk_h1")

    # WORRY
    st.markdown("<div class='feature-box'><span class='question-label'>Worried about getting sick from vaccine</span></div>", unsafe_allow_html=True)
    op_h1n1_sick = st.selectbox("", ["Not at all worried","Not very worried","Don't know","Somewhat worried","Very worried"], key="sick_h1")

    # ENCODE INPUT
    doctor_recc_h1n1 = 1.0 if doctor_recc_h1n1 == "Yes" else 0.0
    health_insurance_h1 = 1.0 if health_insurance_h1 == "Yes" else 0.0
    health_worker_h1 = 1.0 if health_worker_h1 == "Yes" else 0.0
    sex_Male_h1 = True if sex_h1 == "Male" else False
    age_encoded_h1 = ["18-34","35-44","45-54","55-64","65+"].index(age_h1) + 1
    education_encoded_h1 = ["Unknown","< 12 Years","12 Years","Some College","College Graduate"].index(education_h1)

    h1n1_input = np.array([[doctor_recc_h1n1,
                            health_insurance_h1,
                            encode5(op_h1n1_effective),
                            encode5(op_h1n1_risk),
                            encode5(op_h1n1_sick),
                            age_encoded_h1,
                            sex_Male_h1,
                            health_worker_h1,
                            education_encoded_h1]])

    st.write("")
    if st.button("Predict H1N1", key="pred_h1"):
        prob = h1n1_model.predict_proba(h1n1_input)[0][1]
        st.markdown(f"<div class='result-box'>Probability: {prob:.2%}</div>", unsafe_allow_html=True)


########################################################
# MIDDLE SPACER
########################################################
with spacer:
    st.write("")


########################################################
# RIGHT PANEL — SEASONAL MODEL
########################################################
with right:
    st.subheader("Seasonal Model")

    # SEX (ignored)
    st.markdown("<div class='feature-box'><span class='question-label'>Sex</span></div>", unsafe_allow_html=True)
    sex_fake = st.radio("", ["Male", "Female"], key="sex_seas")

    # AGE
    st.markdown("<div class='feature-box'><span class='question-label'>Age group</span></div>", unsafe_allow_html=True)
    age_seas = st.selectbox("", ["18-34","35-44","45-54","55-64","65+"], key="age_seas")

    # EDUCATION
    st.markdown("<div class='feature-box'><span class='question-label'>Education</span></div>", unsafe_allow_html=True)
    education_seas = st.selectbox("", ["Unknown","< 12 Years","12 Years","Some College","College Graduate"], key="edu_seas")

    # HEALTH INSURANCE
    st.markdown("<div class='feature-box'><span class='question-label'>Health insurance</span></div>", unsafe_allow_html=True)
    health_insurance_seas = st.radio("", ["Yes", "No"], key="ins_seas")

    # HEALTH WORKER
    st.markdown("<div class='feature-box'><span class='question-label'>Health worker</span></div>", unsafe_allow_html=True)
    health_worker_seas = st.radio("", ["Yes", "No"], key="hw_seas")

    # DOCTOR RECOMMENDATION
    st.markdown("<div class='feature-box'><span class='question-label'>Doctor recommendation</span></div>", unsafe_allow_html=True)
    doctor_recc_seas = st.radio("", ["Yes", "No"], key="dr_seas")

    # EFFECTIVENESS
    st.markdown("<div class='feature-box'><span class='question-label'>Vaccine perceived effectiveness</span></div>", unsafe_allow_html=True)
    op_seas_effective = st.selectbox("", ["Not at all effective","Not very effective","Don't know","Somewhat effective","Very effective"], key="eff_seas")

    # RISK
    st.markdown("<div class='feature-box'><span class='question-label'>Risk if unvaccinated</span></div>", unsafe_allow_html=True)
    op_seas_risk = st.selectbox("", ["Very low","Somewhat low","Don't know","Somewhat high","Very high"], key="risk_seas")

    # WORRY
    st.markdown("<div class='feature-box'><span class='question-label'>Worried about getting sick from vaccine</span></div>", unsafe_allow_html=True)
    op_seas_sick = st.selectbox("", ["Not at all worried","Not very worried","Don't know","Somewhat worried","Very worried"], key="sick_seas")

    # ENCODE INPUT
    doctor_recc_seas = 1.0 if doctor_recc_seas == "Yes" else 0.0
    health_insurance_seas = 1.0 if health_insurance_seas == "Yes" else 0.0
    health_worker_seas = 1.0 if health_worker_seas == "Yes" else 0.0
    age_encoded_seas = ["18-34","35-44","45-54","55-64","65+"].index(age_seas) + 1
    education_encoded_seas = ["Unknown","< 12 Years","12 Years","Some College","College Graduate"].index(education_seas)

    seasonal_input = np.array([[doctor_recc_seas,
                                age_encoded_seas,
                                health_insurance_seas,
                                encode5(op_seas_effective),
                                encode5(op_seas_risk),
                                encode5(op_seas_sick),
                                health_worker_seas,
                                education_encoded_seas]])

    st.write("")
    if st.button("Predict Seasonal", key="pred_seas"):
        prob = seasonal_model.predict_proba(seasonal_input)[0][1]
        st.markdown(f"<div class='result-box'>Probability: {prob:.2%}</div>", unsafe_allow_html=True)
