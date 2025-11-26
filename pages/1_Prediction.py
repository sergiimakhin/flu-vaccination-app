import streamlit as st
import pickle
import numpy as np

# Load models
with open("h1n1_model.pkl", "rb") as f:
    h1n1_model = pickle.load(f)

with open("seasonal_model.pkl", "rb") as f:
    seasonal_model = pickle.load(f)

st.title("Vaccine Uptake Prediction")

# STYLE — adds spacing and separators
st.markdown("""
<style>
.block {padding-top: 8px; padding-bottom: 8px;}
.sep {border-bottom: 1px solid #999; margin-bottom: 12px;}
</style>
""", unsafe_allow_html=True)

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
}.get(x, 3)

#############################################
# THREE COLUMNS: left panel + empty space + right panel
#############################################
left, spacer, right = st.columns([1.15, 0.35, 1])   

################################################
# LEFT PANEL — H1N1 MODEL
################################################
with left:
    st.subheader("H1N1 Model")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    sex_h1 = st.radio("Sex", ["Male", "Female"], key="sex_h1")
    
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    age_h1 = st.selectbox("Age group", [
        "18-34", "35-44", "45-54", "55-64", "65+"
    ], key="age_h1")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    education_h1 = st.selectbox("Education", [
        "Unknown", "< 12 Years", "12 Years", "Some College", "College Graduate"
    ], key="edu_h1")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    health_insurance_h1 = st.radio("Health insurance", ["Yes", "No"], key="ins_h1")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    health_worker_h1 = st.radio("Healthcare worker", ["Yes", "No"], key="hw_h1")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    doctor_recc_h1n1 = st.radio("Doctor recommendation", ["Yes", "No"], key="dr_h1")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    op_h1n1_effective = st.selectbox(
        "Opinion: vaccine effectiveness",
        ["Not at all effective", "Not very effective", "Don't know",
         "Somewhat effective", "Very effective"], key="eff_h1"
    )

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    op_h1n1_risk = st.selectbox(
        "Opinion: risk if unvaccinated",
        ["Very low", "Somewhat low", "Don't know",
         "Somewhat high", "Very high"], key="risk_h1"
    )

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    op_h1n1_sick = st.selectbox(
        "Worried about getting sick from the vaccine",
        ["Not at all worried", "Not very worried", "Don't know",
         "Somewhat worried", "Very worried"], key="sick_h1"
    )

    # ENCODE
    doctor_recc_h1n1 = 1.0 if doctor_recc_h1n1 == "Yes" else 0.0
    health_insurance_h1 = 1.0 if health_insurance_h1 == "Yes" else 0.0
    health_worker_h1 = 1.0 if health_worker_h1 == "Yes" else 0.0
    sex_Male_h1 = True if sex_h1 == "Male" else False

    age_encoded_h1 = ["18-34", "35-44", "45-54", "55-64", "65+"].index(age_h1) + 1
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

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    if st.button("Predict H1N1", key="pred_h1"):
        prob = h1n1_model.predict_proba(h1n1_input)[0][1]
        st.success(f"H1N1 vaccine probability: {prob:.2%}")
        st.progress(float(prob))


################################################
# MIDDLE COLUMN = EMPTY BLANK SPACE
################################################
with spacer:
    st.write("")   # intentionally left blank
    st.write("")   # extra spacing control


################################################
# RIGHT PANEL — SEASONAL MODEL
################################################
with right:
    st.subheader("Seasonal Model")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    sex_fake = st.radio("Sex", ["Male", "Female"], key="sex_seas")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    age_seas = st.selectbox("Age group", [
        "18-34", "35-44", "45-54", "55-64", "65+"
    ], key="age_seas")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    education_seas = st.selectbox("Education", [
        "Unknown", "< 12 Years", "12 Years", "Some College", "College Graduate"
    ], key="edu_seas")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    health_insurance_seas = st.radio("Health insurance", ["Yes", "No"], key="ins_seas")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    health_worker_seas = st.radio("Healthcare worker", ["Yes", "No"], key="hw_seas")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    doctor_recc_seas = st.radio("Doctor recommendation", ["Yes", "No"], key="dr_seas")

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    op_seas_effective = st.selectbox(
        "Opinion: vaccine effectiveness",
        ["Not at all effective", "Not very effective", "Don't know",
         "Somewhat effective", "Very effective"], key="eff_seas"
    )

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    op_seas_risk = st.selectbox(
        "Opinion: risk if unvaccinated",
        ["Very low", "Somewhat low", "Don't know",
         "Somewhat high", "Very high"], key="risk_seas"
    )

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    op_seas_sick = st.selectbox(
        "Worried about getting sick from the vaccine",
        ["Not at all worried", "Not very worried", "Don't know",
         "Somewhat worried", "Very worried"], key="sick_seas"
    )

    # Encode
    doctor_recc_seas = 1.0 if doctor_recc_seas == "Yes" else 0.0
    health_insurance_seas = 1.0 if health_insurance_seas == "Yes" else 0.0
    health_worker_seas = 1.0 if health_worker_seas == "Yes" else 0.0

    age_encoded_seas = ["18-34", "35-44", "45-54", "55-64", "65+"].index(age_seas) + 1
    education_encoded_seas = ["Unknown","< 12 Years","12 Years","Some College","College Graduate"].index(education_seas)

    seasonal_input = np.array([[doctor_recc_seas,
                                age_encoded_seas,
                                health_insurance_seas,
                                encode5(op_seas_effective),
                                encode5(op_seas_risk),
                                encode5(op_seas_sick),
                                health_worker_seas,
                                education_encoded_seas]])

    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    if st.button("Predict Seasonal", key="pred_seas"):
        prob = seasonal_model.predict_proba(seasonal_input)[0][1]
        st.success(f"Seasonal vaccine probability: {prob:.2%}")
        st.progress(float(prob))
