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
# GLOBAL STYLE
#############################################
st.markdown(
    """
<style>
body { background-color:#FFFFFF; color:#000000; }

h1, h2, h3, h4 {
    color:#000000;
}

.q-label-blue {
    background-color:#CDE6EE;
    padding:4px 8px;
    border-radius:4px;
    font-weight:600;
    display:inline-block;
    margin-bottom:2px;
}
.q-label-white {
    background-color:#FFFFFF;
    padding:4px 8px;
    border-radius:4px;
    font-weight:600;
    border:1px solid #CDE6EE;
    display:inline-block;
    margin-bottom:2px;
}

.result-box {
    background-color:#CDE6EE;
    padding:10px;
    border-radius:5px;
    margin-top:6px;
}
</style>
""",
    unsafe_allow_html=True,
)

#############################################
# TITLE
#############################################
st.markdown(
    "<h1 style='text-align:center; margin-bottom:0.5rem;'>Vaccine Uptake Prediction</h1>",
    unsafe_allow_html=True,
)
st.write("")

#############################################
# LAYOUT: TWO PANELS + SPACER
#############################################
left, spacer, right = st.columns([1.15, 0.4, 1])

########################################################
# LEFT PANEL — H1N1 MODEL
########################################################
with left:
    st.subheader("H1N1 Model")

    # Sex (blue)
    st.markdown("<span class='q-label-blue'>Sex</span>", unsafe_allow_html=True)
    sex_h1 = st.radio("", ["Male", "Female"], key="sex_h1")

    # Age (white)
    st.markdown("<span class='q-label-white'>Age group</span>", unsafe_allow_html=True)
    age_h1 = st.selectbox(
        "",
        ["18-34", "35-44", "45-54", "55-64", "65+"],
        key="age_h1",
    )

    # Education (blue)
    st.markdown("<span class='q-label-blue'>Education</span>", unsafe_allow_html=True)
    education_h1 = st.selectbox(
        "",
        ["Unknown", "< 12 Years", "12 Years", "Some College", "College Graduate"],
        key="edu_h1",
    )

    # Health insurance (white)
    st.markdown("<span class='q-label-white'>Health insurance</span>", unsafe_allow_html=True)
    health_insurance_h1 = st.radio("", ["Yes", "No"], key="ins_h1")

    # Health worker (blue)
    st.markdown("<span class='q-label-blue'>Health worker</span>", unsafe_allow_html=True)
    health_worker_h1 = st.radio("", ["Yes", "No"], key="hw_h1")

    # Doctor recommendation (white)
    st.markdown("<span class='q-label-white'>Doctor recommendation</span>", unsafe_allow_html=True)
    doctor_recc_h1n1 = st.radio("", ["Yes", "No"], key="dr_h1")

    # Effectiveness (blue)
    st.markdown(
        "<span class='q-label-blue'>Vaccine perceived effectiveness</span>",
        unsafe_allow_html=True,
    )
    op_h1n1_effective = st.selectbox(
        "",
        [
            "Not at all effective",
            "Not very effective",
            "Don't know",
            "Somewhat effective",
            "Very effective",
        ],
        key="eff_h1",
    )

    # Risk (white)
    st.markdown("<span class='q-label-white'>Risk if unvaccinated</span>", unsafe_allow_html=True)
    op_h1n1_risk = st.selectbox(
        "",
        ["Very low", "Somewhat low", "Don't know", "Somewhat high", "Very high"],
        key="risk_h1",
    )

    # Worry (blue)
    st.markdown(
        "<span class='q-label-blue'>Worried about getting sick from vaccine</span>",
        unsafe_allow_html=True,
    )
    op_h1n1_sick = st.selectbox(
        "",
        [
            "Not at all worried",
            "Not very worried",
            "Don't know",
            "Somewhat worried",
            "Very worried",
        ],
        key="sick_h1",
    )

    # ENCODE INPUT
    doctor_recc_h1n1 = 1.0 if doctor_recc_h1n1 == "Yes" else 0.0
    health_insurance_h1 = 1.0 if health_insurance_h1 == "Yes" else 0.0
    health_worker_h1 = 1.0 if health_worker_h1 == "Yes" else 0.0
    sex_Male_h1 = True if sex_h1 == "Male" else False
    age_encoded_h1 = ["18-34", "35-44", "45-54", "55-64", "65+"].index(age_h1) + 1
    education_encoded_h1 = [
        "Unknown",
        "< 12 Years",
        "12 Years",
        "Some College",
        "College Graduate",
    ].index(education_h1)

    h1n1_input = np.array(
        [
            [
                doctor_recc_h1n1,
                health_insurance_h1,
                encode5(op_h1n1_effective),
                encode5(op_h1n1_risk),
                encode5(op_h1n1_sick),
                age_encoded_h1,
                sex_Male_h1,
                health_worker_h1,
                education_encoded_h1,
            ]
        ]
    )

    st.write("")
    if st.button("Predict H1N1", key="pred_h1"):
        prob = h1n1_model.predict_proba(h1n1_input)[0][1]
        st.markdown(
            f"<div class='result-box'>Probability of H1N1 vaccination: "
            f"<strong>{prob:.2%}</strong></div>",
            unsafe_allow_html=True,
        )

########################################################
# SPACER
########################################################
with spacer:
    st.write("")

########################################################
# RIGHT PANEL — SEASONAL MODEL
########################################################
with right:
    st.subheader("Seasonal Model")

    # Sex (fake, blue)
    st.markdown("<span class='q-label-blue'>Sex</span>", unsafe_allow_html=True)
    sex_fake = st.radio("", ["Male", "Female"], key="sex_seas")  # ignored

    # Age (white)
    st.markdown("<span class='q-label-white'>Age group</span>", unsafe_allow_html=True)
    age_seas = st.selectbox(
        "",
        ["18-34", "35-44", "45-54", "55-64", "65+"],
        key="age_seas",
    )

    # Education (blue)
    st.markdown("<span class='q-label-blue'>Education</span>", unsafe_allow_html=True)
    education_seas = st.selectbox(
        "",
        ["Unknown", "< 12 Years", "12 Years", "Some College", "College Graduate"],
        key="edu_seas",
    )

    # Health insurance (white)
    st.markdown("<span class='q-label-white'>Health insurance</span>", unsafe_allow_html=True)
    health_insurance_seas = st.radio("", ["Yes", "No"], key="ins_seas")

    # Health worker (blue)
    st.markdown("<span class='q-label-blue'>Health worker</span>", unsafe_allow_html=True)
    health_worker_seas = st.radio("", ["Yes", "No"], key="hw_seas")

    # Doctor recommendation (white)
    st.markdown("<span class='q-label-white'>Doctor recommendation</span>", unsafe_allow_html=True)
    doctor_recc_seas = st.radio("", ["Yes", "No"], key="dr_seas")

    # Effectiveness (blue)
    st.markdown(
        "<span class='q-label-blue'>Vaccine perceived effectiveness</span>",
        unsafe_allow_html=True,
    )
    op_seas_effective = st.selectbox(
        "",
        [
            "Not at all effective",
            "Not very effective",
            "Don't know",
            "Somewhat effective",
            "Very effective",
        ],
        key="eff_seas",
    )

    # Risk (white)
    st.markdown("<span class='q-label-white'>Risk if unvaccinated</span>", unsafe_allow_html=True)
    op_seas_risk = st.selectbox(
        "",
        ["Very low", "Somewhat low", "Don't know", "Somewhat high", "Very high"],
        key="risk_seas",
    )

    # Worry (blue)
    st.markdown(
        "<span class='q-label-blue'>Worried about getting sick from vaccine</span>",
        unsafe_allow_html=True,
    )
    op_seas_sick = st.selectbox(
        "",
        [
            "Not at all worried",
            "Not very worried",
            "Don't know",
            "Somewhat worried",
            "Very worried",
        ],
        key="sick_seas",
    )

    # ENCODE INPUT
    doctor_recc_seas = 1.0 if doctor_recc_seas == "Yes" else 0.0
    health_insurance_seas = 1.0 if health_insurance_seas == "Yes" else 0.0
    health_worker_seas = 1.0 if health_worker_seas == "Yes" else 0.0
    age_encoded_seas = ["18-34", "35-44", "45-54", "55-64", "65+"].index(age_seas) + 1
    education_encoded_seas = [
        "Unknown",
        "< 12 Years",
        "12 Years",
        "Some College",
        "College Graduate",
    ].index(education_seas)

    seasonal_input = np.array(
        [
            [
                doctor_recc_seas,
                age_encoded_seas,
                health_insurance_seas,
                encode5(op_seas_effective),
                encode5(op_seas_risk),
                encode5(op_seas_sick),
                health_worker_seas,
                education_encoded_seas,
            ]
        ]
    )

    st.write("")
    if st.button("Predict Seasonal", key="pred_seas"):
        prob = seasonal_model.predict_proba(seasonal_input)[0][1]
        st.markdown(
            f"<div class='result-box'>Probability of Seasonal vaccination: "
            f"<strong>{prob:.2%}</strong></div>",
            unsafe_allow_html=True,
        )
