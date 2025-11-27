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
# LOAD FONT + STYLE
#############################################
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Assistant:wght@300;400;600;700&display=swap');

html, body, div, p, span, label {
    font-family: 'Assistant', sans-serif !important;
    font-weight:400;
}

.block-panel {
    background-color:#58ADC5;
    padding:8px;
    border-radius:12px;
    margin-bottom:8px;
}

.question-label {
    font-weight:600;
    font-size:18px;
}

.result-box {
    background-color:#F08C66;
    color:#000000;
    padding:14px;
    border-radius:12px;
    font-size:20px;
    font-weight:700;
    margin-top:10px;
    border: 1px solid #EA5B25;
}

.top-border {
    border-top:3px solid #8AC6D6;
    padding-top:10px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)


#############################################
# TITLE
#############################################
st.markdown(
    "<h1 style='text-align:center; margin-bottom:1rem;'>Vaccine Uptake Prediction</h1>",
    unsafe_allow_html=True,
)


#############################################
# 3-COLUMN LAYOUT
#############################################
left, spacer, right = st.columns([1.5, 0.3, 1.5])


#############################################################
# LEFT PANEL — H1N1
#############################################################
with left:
    st.markdown("<div class='top-border'></div>", unsafe_allow_html=True)
    st.subheader("H1N1 Model")

    st.markdown("<div class='block-panel'><span class='question-label'>Sex</span></div>", unsafe_allow_html=True)
    sex_h1 = st.radio("", ["Male", "Female"], key="sex_h1")

    st.markdown("<div class='block-panel'><span class='question-label'>Age group</span></div>", unsafe_allow_html=True)
    age_h1 = st.selectbox("", ["18-34","35-44","45-54","55-64","65+"], key="age_h1")

    st.markdown("<div class='block-panel'><span class='question-label'>Education</span></div>", unsafe_allow_html=True)
    education_h1 = st.selectbox("", ["Unknown","< 12 Years","12 Years","Some College","College Graduate"], key="edu_h1")

    st.markdown("<div class='block-panel'><span class='question-label'>Health insurance</span></div>", unsafe_allow_html=True)
    health_insurance_h1 = st.radio("", ["Yes", "No"], key="ins_h1")

    st.markdown("<div class='block-panel'><span class='question-label'>Health worker</span></div>", unsafe_allow_html=True)
    health_worker_h1 = st.radio("", ["Yes", "No"], key="hw_h1")

    st.markdown("<div class='block-panel'><span class='question-label'>Doctor recommendation</span></div>", unsafe_allow_html=True)
    doctor_recc_h1n1 = st.radio("", ["Yes", "No"], key="dr_h1")

    st.markdown("<div class='block-panel'><span class='question-label'>Vaccine perceived effectiveness</span></div>", unsafe_allow_html=True)
    op_h1n1_effective = st.selectbox("", ["Not at all effective","Not very effective","Don't know","Somewhat effective","Very effective"], key="eff_h1")

    st.markdown("<div class='block-panel'><span class='question-label'>Risk if unvaccinated</span></div>", unsafe_allow_html=True)
    op_h1n1_risk = st.selectbox("", ["Very low","Somewhat low","Don't know","Somewhat high","Very high"], key="risk_h1")

    st.markdown("<div class='block-panel'><span class='question-label'>Worried about getting sick from vaccine</span></div>", unsafe_allow_html=True)
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


#############################################################
# SPACER
#############################################################
with spacer:
    st.write("")


#############################################################
# RIGHT PANEL — SEASONAL
#############################################################
with right:
    st.markdown("<div class='top-border'></div>", unsafe_allow_html=True)
    st.subheader("Seasonal Model")

    st.markdown("<div class='block-panel'><span class='question-label'>Sex</span></div>", unsafe_allow_html=True)
    sex_fake = st.radio("", ["Male", "Female"], key="sex_seas")

    st.markdown("<div class='block-panel'><span class='question-label'>Age group</span></div>", unsafe_allow_html=True)
    age_seas = st.selectbox("", ["18-34","35-44","45-54","55-64","65+"], key="age_seas")

    st.markdown("<div class='block-panel'><span class='question-label'>Education</span></div>", unsafe_allow_html=True)
    education_seas = st.selectbox("", ["Unknown","< 12 Years","12 Years","Some College","College Graduate"], key="edu_seas")

    st.markdown("<div class='block-panel'><span class='question-label'>Health insurance</span></div>", unsafe_allow_html=True)
    health_insurance_seas = st.radio("", ["Yes", "No"], key="ins_seas")

    st.markdown("<div class='block-panel'><span class='question-label'>Health worker</span></div>", unsafe_allow_html=True)
    health_worker_seas = st.radio("", ["Yes", "No"], key="hw_seas")

    st.markdown("<div class='block-panel'><span class='question-label'>Doctor recommendation</span></div>", unsafe_allow_html=True)
    doctor_recc_seas = st.radio("", ["Yes", "No"], key="dr_seas")

    st.markdown("<div class='block-panel'><span class='question-label'>Vaccine perceived effectiveness</span></div>", unsafe_allow_html=True)
    op_seas_effective = st.selectbox("", ["Not at all effective","Not very effective","Don't know","Somewhat effective","Very effective"], key="eff_seas")

    st.markdown("<div class='block-panel'><span class='question-label'>Risk if unvaccinated</span></div>", unsafe_allow_html=True)
    op_seas_risk = st.selectbox("", ["Very low","Somewhat low","Don't know","Somewhat high","Very high"], key="risk_seas")

    st.markdown("<div class='block-panel'><span class='question-label'>Worried about getting sick from vaccine</span></div>", unsafe_allow_html=True)
    op_seas_sick = st.selectbox("", ["Not at all worried","Not very worried","Don't know","Somewhat worried","Very worried"], key="sick_seas")

    # ENCODE
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
