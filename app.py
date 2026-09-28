import joblib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)


# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_artifacts():
    model = load_model("heart_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


model, scaler = load_artifacts()


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0b0e14;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---------------- FORM CARD ---------------- */
    div[data-testid="stForm"] {
        background: #12161f;
        border: 1px solid #232838;
        border-radius: 18px;
        padding: 2rem 2.2rem 1.4rem 2.2rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.45);
    }

    /* ---------------- SECTION LABELS ---------------- */
    .field-group {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: #ff8a8a;
        margin: 0.2rem 0 0.9rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #232838;
    }

    /* ---------------- LABELS ---------------- */
    label[data-testid="stWidgetLabel"] p {
        font-weight: 500;
        font-size: 0.92rem;
        color: #b8beca;
        margin-bottom: 0.15rem;
    }

    /* ---------------- NUMBER / TEXT INPUTS ---------------- */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        background-color: #1a2030 !important;
        border: 1.5px solid #2e3648 !important;
        border-radius: 10px !important;
        color: #f1f3f8 !important;
        font-weight: 500;
        font-size: 0.98rem;
        padding: 0.55rem 0.7rem !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }

    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus {
        border-color: #ff5c5c !important;
        box-shadow: 0 0 0 3px rgba(255, 92, 92, 0.18) !important;
    }

    div[data-testid="stNumberInput"] button {
        background-color: #1a2030 !important;
        border: 1.5px solid #2e3648 !important;
    }
    div[data-testid="stNumberInput"] button:hover {
        border-color: #ff5c5c !important;
    }

    /* ---------------- SELECT BOXES ---------------- */
    div[data-baseweb="select"] > div {
        background-color: #1a2030 !important;
        border: 1.5px solid #2e3648 !important;
        border-radius: 10px !important;
        color: #f1f3f8 !important;
        font-weight: 500;
        min-height: 44px;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #454e63 !important;
    }
    div[data-baseweb="select"]:focus-within > div {
        border-color: #ff5c5c !important;
        box-shadow: 0 0 0 3px rgba(255, 92, 92, 0.18) !important;
    }

    /* dropdown popover menu */
    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #1a2030 !important;
        border: 1px solid #2e3648 !important;
        border-radius: 10px !important;
    }
    li[data-testid="stSelectboxVirtualDropdown"],
    ul[data-testid="stSelectboxVirtualDropdown"] li {
        color: #f1f3f8 !important;
    }
    ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
        background-color: #262d40 !important;
    }

    /* ---------------- BUTTON ---------------- */
    .stFormSubmitButton > button {
        background: linear-gradient(120deg, #ff5c5c, #e63946);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 8px 24px rgba(230, 57, 70, 0.35);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-top: 0.6rem;
    }
    .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(230, 57, 70, 0.5);
        color: white;
        border: none;
    }
    .stFormSubmitButton > button:active {
        transform: translateY(0px);
    }

    /* ---------------- RESULT CARD ---------------- */
    .result-box {
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 1.4rem;
        border: 1.5px solid;
    }
    .result-positive {
        background: rgba(248, 113, 113, 0.10);
        color: #ff8080;
        border-color: rgba(248, 113, 113, 0.45);
        box-shadow: 0 0 30px rgba(248, 113, 113, 0.12);
    }
    .result-negative {
        background: rgba(52, 211, 153, 0.10);
        color: #4ade9c;
        border-color: rgba(52, 211, 153, 0.45);
        box-shadow: 0 0 30px rgba(52, 211, 153, 0.12);
    }
    .result-sub {
        font-size: 0.95rem;
        font-weight: 400;
        color: #9aa3b5;
        margin-top: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    "<h1 style='text-align:center; color:#ff5c5c; letter-spacing:-0.5px; margin-bottom:0;'>"
    "❤️ Heart Disease Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; color:#8890a0; margin-top:0.3rem;'>"
    "AI-Based Medical Prediction </p>",
    unsafe_allow_html=True
)
st.write("")


# ============================================================
# MAPPINGS (must match training encoding exactly)
# ============================================================

CP_MAP = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

FBS_MAP = {"No": 0, "Yes": 1}

RESTECG_MAP = {
    "Normal": 0,
    "ST-T Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}

EXANG_MAP = {"No": 0, "Yes": 1}

SLOPE_MAP = {
    "Upsloping": 0,
    "Flat": 1,
    "Downsloping": 2
}

THAL_MAP = {
    "Normal": 1,
    "Fixed Defect": 2,
    "Reversible Defect": 3
}


# ============================================================
# INPUT FORM
# ============================================================

with st.form("patient_form"):

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="field-group">Patient Information</div>', unsafe_allow_html=True)
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        gender = st.selectbox("Gender", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", list(CP_MAP.keys()))
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=300, value=120)
        chol = st.number_input("Cholesterol", min_value=0, max_value=700, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", list(FBS_MAP.keys()))
        restecg = st.selectbox("Resting ECG Result", list(RESTECG_MAP.keys()))

    with col2:
        st.markdown('<div class="field-group">Clinical Measurements</div>', unsafe_allow_html=True)
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", list(EXANG_MAP.keys()))
        oldpeak = st.number_input("Old Peak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST", list(SLOPE_MAP.keys()))
        ca = st.selectbox("Number of Major Vessels", ["0", "1", "2", "3", "4"])
        thal = st.selectbox("Thalassemia", list(THAL_MAP.keys()))

    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    try:
        sex = 1 if gender == "Male" else 0

        patient = np.array([[
            float(age),
            sex,
            CP_MAP[cp],
            float(trestbps),
            float(chol),
            FBS_MAP[fbs],
            RESTECG_MAP[restecg],
            float(thalach),
            EXANG_MAP[exang],
            float(oldpeak),
            SLOPE_MAP[slope],
            int(ca),
            THAL_MAP[thal]
        ]])

        patient_scaled = scaler.transform(patient)

        prediction = model.predict(patient_scaled, verbose=0)
        probability = float(prediction[0][0])

        if probability >= 0.5:
            st.markdown(
                f"<div class='result-box result-positive'>🔴 Heart Disease Detected"
                f"<div class='result-sub'>Confidence: {probability * 100:.1f}%</div></div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='result-box result-negative'>🟢 No Heart Disease Detected"
                f"<div class='result-sub'>Confidence: {(1 - probability) * 100:.1f}%</div></div>",
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"Something went wrong: {e}")


st.write("")
st.markdown(
    "<p style='text-align:center; color:#5c6270; font-size:0.85rem;'>"
    "An AI-powered tool to analyze health indicators and predict the likelihood of heart"
    "</p>",
    unsafe_allow_html=True
)
