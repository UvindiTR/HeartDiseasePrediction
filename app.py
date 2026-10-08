import base64
import os

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
    layout="wide"
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
# PAGE STATE (form page -> result page)
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "form"
    st.session_state.result = None


# ============================================================
# APPEARANCE SETTINGS (theme, font size, background photo)
# ============================================================

THEMES = {
    "Dark": {
        "bg": "#0b0e14", "bg_grad": "radial-gradient(circle at 20% 0%, #1a1020 0%, #0b0e14 45%)",
        "overlay": "rgba(8,10,16,0.72)", "card": "rgba(18,22,31,0.93)", "solid": "#12161f",
        "card2": "#1a2030", "border": "#2e3648", "text": "#f1f3f8", "muted": "#aab2c2",
        "accent": "#ff5c5c", "accent_text": "#ff8a8a", "track": "#232838",
        "good": "#4ade9c", "bad": "#ff6b6b", "warn": "#facc15", "side": "#0f131c",
    },
    "Light": {
        "bg": "#f4f6fb", "bg_grad": "linear-gradient(160deg, #fdf2f3 0%, #f4f6fb 50%)",
        "overlay": "rgba(255,255,255,0.68)", "card": "rgba(255,255,255,0.95)", "solid": "#ffffff",
        "card2": "#eef1f7", "border": "#d5dbe8", "text": "#1b2233", "muted": "#5b6578",
        "accent": "#e63946", "accent_text": "#d62839", "track": "#e3e8f2",
        "good": "#15935f", "bad": "#d62839", "warn": "#c79100", "side": "#ffffff",
    },
}
FONT_PX = {"Small": 14, "Medium": 16, "Large": 22}


def load_background(uploaded):
    """Return (mime, base64) for the uploaded photo, or a local background file if present."""
    if uploaded is not None:
        if uploaded.size > 4 * 1024 * 1024:
            st.sidebar.warning(
                "Photo is larger than 4 MB. Please use a smaller image.")
            return None
        return uploaded.type, base64.b64encode(uploaded.getvalue()).decode()
    here = os.path.dirname(os.path.abspath(__file__))
    for name, mime in (("background.jpg", "image/jpeg"), ("background.jpeg", "image/jpeg"), ("background.png", "image/png")):
        # looks next to app.py, wherever Streamlit was started
        path = os.path.join(here, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                return mime, base64.b64encode(f.read()).decode()
    return None


with st.sidebar:
    st.markdown("<div class='side-title'>Appearance</div>",
                unsafe_allow_html=True)
    theme_name = st.radio("Theme", ["Dark", "Light"], horizontal=True)

size_name = "Large"   # fixed font size
# background comes from background.jpg / background.png in the project folder
photo = None

t = THEMES[theme_name]
bg = load_background(photo)


# ============================================================
# STYLING
# ============================================================

if bg:
    bg_rule = (
        f".stApp {{background-image: linear-gradient({t['overlay']}, {t['overlay']}), "
        f"url(data:{bg[0]};base64,{bg[1]}); background-size: cover; "
        f"background-position: center; background-attachment: fixed;}}"
    )
else:
    bg_rule = f".stApp {{background: {t['bg_grad']};}}"

st.markdown(
    f"""
    <style>
    :root {{
        --bg:{t['bg']}; --card:{t['card']}; --solid:{t['solid']}; --card2:{t['card2']};
        --border:{t['border']}; --text:{t['text']}; --muted:{t['muted']};
        --accent:{t['accent']}; --accent-text:{t['accent_text']}; --track:{t['track']};
        --good:{t['good']}; --bad:{t['bad']}; --warn:{t['warn']};
    }}
    html {{font-size: {FONT_PX[size_name]}px !important;}}
    {bg_rule}
    section[data-testid="stSidebar"] {{background: {t['side']};}}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    .block-container {max-width: 1250px; padding-top: 2rem;}
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}

    /* ---------- TOP BAR: no dark strip in light mode ---------- */
    header[data-testid="stHeader"] {background: transparent !important;}
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {background: transparent !important;}

    /* ---------- TEXT COLOURS (work in light and dark) ---------- */
    label[data-testid="stWidgetLabel"] p,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label div,
    div[data-testid="stFileUploader"] label p,
    div[data-testid="stFileUploader"] small,
    div[data-testid="stFileUploader"] span {color: var(--text) !important;}
    label[data-testid="stWidgetLabel"] p {font-weight: 500; font-size: 1.05rem;}

    /* ---------- SIDEBAR (wider, larger text) ---------- */
    section[data-testid="stSidebar"] {border-right: 1px solid var(--border);
        width: 440px !important; min-width: 440px !important;}
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {font-size: 1.1rem; font-weight: 600;}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label p {font-size: 1.05rem;}
    .side-title {color: var(--accent-text); font-size: 0.95rem; font-weight: 700;
                 letter-spacing: 0.09em; text-transform: uppercase; margin: 1.4rem 0 0.6rem 0;}
    .side-text {color: var(--muted); font-size: 1.05rem; line-height: 1.65;}

    /* more room in the sidebar, especially the Appearance part */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {padding: 2.5rem 2rem 2rem 2rem;}
    section[data-testid="stSidebar"] div[data-testid="stRadio"] {margin: 0.6rem 0 1.6rem 0;}
    section[data-testid="stSidebar"] div[role="radiogroup"] {gap: 1.6rem;}
    section[data-testid="stSidebar"] .side-title:first-child {margin-top: 0;}

    /* file uploader box (was dark in light mode) */
    div[data-testid="stFileUploaderDropzone"] {background: var(--card2) !important; border: 1.5px dashed var(--border) !important;
        border-radius: 12px !important;}
    div[data-testid="stFileUploaderDropzone"] button {background: var(--solid) !important; color: var(--text) !important;
        border: 1.5px solid var(--border) !important;}
    div[data-testid="stFileUploaderDropzone"] * {color: var(--text) !important;}

    /* ---------- HERO ---------- */
    .hero {background: var(--card); border: 1px solid var(--border); border-radius: 20px;
           padding: 2rem 2.2rem; margin-bottom: 1.6rem; text-align: center;}
    .hero-title {color: var(--accent) !important; font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px;}
    .hero-sub {color: var(--muted); font-size: 1.1rem; margin-top: 0.4rem;}
    .chips {margin-top: 1.1rem;}
    .chip {display: inline-block; background: var(--card2); border: 1px solid var(--border);
           color: var(--text); border-radius: 999px; padding: 0.35rem 0.9rem; font-size: 0.9rem; margin: 0.2rem;}

    /* ---------- FORM: outer box invisible, two separate cards ---------- */
    div[data-testid="stForm"] {background: transparent; border: none; padding: 0;}
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card); border: 1px solid var(--border) !important;
        border-radius: 18px; padding: 1rem 1.2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.12);}
    .field-group {font-size: 0.9rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
        color: var(--accent-text); margin: 0.2rem 0 0.9rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);}

    div[data-testid="stNumberInput"] input {
        background-color: var(--card2) !important; border: 1.5px solid var(--border) !important;
        border-radius: 10px !important; color: var(--text) !important; font-weight: 500;
        font-size: 1.05rem; padding: 0.6rem 0.7rem !important;}
    div[data-testid="stNumberInput"] input:focus {
        border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(255,92,92,0.18) !important;}
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {background-color: var(--card2) !important;}
    div[data-testid="stNumberInput"] button {background-color: var(--card2) !important;
        border: 1.5px solid var(--border) !important; color: var(--text) !important;}
    div[data-testid="stNumberInput"] button:hover {border-color: var(--accent) !important;}

    /* ---------- DROPDOWNS (were dark in light mode) ---------- */
    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] > div > div {
        background-color: var(--card2) !important; color: var(--text) !important;}
    div[data-baseweb="select"] > div {
        border: 1.5px solid var(--border) !important; border-radius: 10px !important;
        font-weight: 500; font-size: 1.05rem; min-height: 46px;}
    div[data-baseweb="select"] * {color: var(--text) !important;}
    div[data-baseweb="select"] svg {fill: var(--text) !important;}
    div[data-baseweb="select"]:focus-within > div {
        border-color: var(--accent) !important; box-shadow: 0 0 0 3px rgba(255,92,92,0.18) !important;}
    div[data-baseweb="popover"] > div, div[data-baseweb="popover"] ul,
    div[data-baseweb="menu"] {background-color: var(--solid) !important; border-radius: 10px !important;}
    div[data-baseweb="popover"] li, div[data-baseweb="popover"] li * {color: var(--text) !important;}
    div[data-baseweb="popover"] li:hover {background-color: var(--card2) !important;}

    /* ---------- BUTTONS ---------- */
    .stFormSubmitButton > button, div[data-testid="stButton"] > button {
        background: linear-gradient(120deg, #ff5c5c, #e63946); color: white; font-weight: 700;
        font-size: 1.15rem; padding: 0.8rem 1rem; border-radius: 12px; border: none;
        box-shadow: 0 8px 24px rgba(230,57,70,0.35); transition: transform 0.15s ease; margin-top: 0.6rem;}
    .stFormSubmitButton > button:hover, div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px); color: white; border: none;}
    .stFormSubmitButton > button p, div[data-testid="stButton"] > button p {color: white !important;}

    /* ---------- RESULT PAGE ---------- */
    .result-card {background: var(--card); border: 1.5px solid var(--border); border-radius: 18px;
        padding: 1.8rem 1.6rem; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.12); margin-top: 0.4rem;}
    .result-card.positive {border-color: var(--bad);}
    .result-card.negative {border-color: var(--good);}
    .result-title {font-size: 1.6rem; font-weight: 800; color: var(--text);}
    .positive .result-title {color: var(--bad);}
    .negative .result-title {color: var(--good);}
    .gauge {width: 14rem; height: 14rem; border-radius: 50%; margin: 1.4rem auto;
        display: flex; align-items: center; justify-content: center;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), var(--track) 0);}
    .gauge-inner {width: 10.6rem; height: 10.6rem; border-radius: 50%; background: var(--solid);
        display: flex; flex-direction: column; align-items: center; justify-content: center;}
    .gauge-val {font-size: 2rem; font-weight: 800; color: var(--text);}
    .gauge-lbl {font-size: 0.9rem; color: var(--muted);}
    .metrics {display: flex; gap: 0.8rem; margin-top: 0.4rem;}
    .metric {flex: 1; background: var(--card2); border: 1px solid var(--border); border-radius: 12px; padding: 0.8rem 0.4rem;}
    .m-val {font-size: 1.5rem; font-weight: 800; color: var(--text);}
    .m-lbl {font-size: 0.9rem; color: var(--muted);}
    .note {margin-top: 1.1rem; font-size: 0.9rem; color: var(--muted);}

    .sum-card {background: var(--card); border: 1px solid var(--border); border-radius: 18px;
        padding: 1.4rem 1.8rem; margin-top: 1.2rem;}
    .sum-title {font-size: 0.9rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase;
        color: var(--accent-text); margin-bottom: 0.8rem;}
    .sum-grid {display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem 2.5rem;}
    .sum-row {display: flex; justify-content: space-between; padding: 0.45rem 0;
        border-bottom: 1px solid var(--border); font-size: 1rem; color: var(--muted);}
    .sum-row b {color: var(--text);}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR INFORMATION
# ============================================================

with st.sidebar:
    st.markdown(
        "<div class='side-title'>About</div>"
        "<div class='side-text'>An AI-based tool that estimates the likelihood of heart disease "
        "from 13 clinical measurements using a trained neural network.</div>"
        "<div class='side-title'>How to use</div>"
        "<div class='side-text'>1. Fill in both cards.<br>2. Click Predict.<br>3. The result opens on a new page.</div>"
        "<div class='side-title'>Disclaimer</div>"
        "<div class='side-text'>This tool supports screening only. It is not a medical diagnosis. "
        "Always consult a qualified doctor.</div>",
        unsafe_allow_html=True
    )


# ============================================================
# HERO
# ============================================================

st.markdown(
    "<div class='hero'>"
    "<div class='hero-title'>❤️ Heart Disease Prediction System</div>"
    "<div class='hero-sub'>Enter the 13 patient details to estimate the risk of heart disease</div>"
    "<div class='chips'>"
    "<span class='chip'>🧠 Neural network model</span>"
    "<span class='chip'>📋 13 clinical inputs</span>"
    "<span class='chip'>📊 Accuracy 83.4% · AUC 0.92</span>"
    "</div></div>",
    unsafe_allow_html=True
)


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

LEVEL_COLOUR = {"Low": "good", "Moderate": "warn", "High": "bad"}


# ============================================================
# PAGE 1: INPUT FORM
# ============================================================

if st.session_state.page == "form":

    with st.form("patient_form"):

        col1, col2 = st.columns(2, gap="large")

        with col1:
            with st.container(border=True):
                st.markdown(
                    '<div class="field-group">🧍 Patient Information</div>', unsafe_allow_html=True)
                age = st.number_input("Age", min_value=1,
                                      max_value=120, value=50)
                gender = st.selectbox("Gender", ["Male", "Female"])
                cp = st.selectbox("Chest Pain Type", list(CP_MAP.keys()))
                trestbps = st.number_input(
                    "Resting Blood Pressure", min_value=0, max_value=300, value=120)
                chol = st.number_input(
                    "Cholesterol", min_value=0, max_value=700, value=200)
                fbs = st.selectbox(
                    "Fasting Blood Sugar > 120 mg/dl", list(FBS_MAP.keys()))
                restecg = st.selectbox(
                    "Resting ECG Result", list(RESTECG_MAP.keys()))

        with col2:
            with st.container(border=True):
                st.markdown(
                    '<div class="field-group">🩺 Clinical Measurements</div>', unsafe_allow_html=True)
                thalach = st.number_input(
                    "Maximum Heart Rate Achieved", min_value=0, max_value=250, value=150)
                exang = st.selectbox(
                    "Exercise Induced Angina", list(EXANG_MAP.keys()))
                oldpeak = st.number_input(
                    "Old Peak (ST Depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                slope = st.selectbox(
                    "Slope of Peak Exercise ST", list(SLOPE_MAP.keys()))
                ca = st.selectbox("Number of Major Vessels",
                                  ["0", "1", "2", "3", "4"])
                thal = st.selectbox("Thalassemia", list(THAL_MAP.keys()))

        submitted = st.form_submit_button(
            "🔍 Predict", use_container_width=True)

    # ------------------------------------------------------------
    # PREDICTION (runs on submit, then moves to the result page)
    # ------------------------------------------------------------

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

            risk_pct = probability * 100

            if risk_pct < 20:
                level = "Low"
            elif risk_pct < 50:
                level = "Moderate"
            else:
                level = "High"

            if probability >= 0.5:
                title, css_class, conf = "🔴 Heart Disease Detected", "positive", probability * 100
            else:
                title, css_class, conf = "🟢 No Heart Disease Detected", "negative", (
                    1 - probability) * 100

            st.session_state.result = {
                "title": title, "css_class": css_class, "risk_pct": risk_pct,
                "conf": conf, "level": level,
                "inputs": {
                    "Age": age, "Gender": gender, "Chest Pain Type": cp,
                    "Resting Blood Pressure": trestbps, "Cholesterol": chol,
                    "Fasting Blood Sugar > 120": fbs, "Resting ECG": restecg,
                    "Max Heart Rate": thalach, "Exercise Angina": exang,
                    "Old Peak": f"{oldpeak:.2f}", "Slope": slope,
                    "Major Vessels": ca, "Thalassemia": thal,
                },
            }
            st.session_state.page = "result"
            st.rerun()

        except Exception as e:
            st.error(f"Something went wrong: {e}")


# ============================================================
# PAGE 2: RESULT
# ============================================================

else:
    r = st.session_state.result
    color = t[LEVEL_COLOUR[r["level"]]]

    _, centre, _ = st.columns([1, 2, 1])

    with centre:
        st.markdown(
            f"<div class='result-card {r['css_class']}'>"
            f"<div class='result-title'>{r['title']}</div>"
            f"<div class='gauge' style='--p:{r['risk_pct']:.1f}; --c:{color};'>"
            f"<div class='gauge-inner'>"
            f"<div class='gauge-val'>{r['risk_pct']:.1f}%</div>"
            f"<div class='gauge-lbl'>Disease probability</div>"
            f"</div></div>"
            f"<div class='metrics'>"
            f"<div class='metric'><div class='m-val'>{r['conf']:.1f}%</div><div class='m-lbl'>Confidence</div></div>"
            f"<div class='metric'><div class='m-val' style='color:{color};'>{r['level']}</div><div class='m-lbl'>Risk level</div></div>"
            f"</div>"
            f"<div class='note'>This tool supports screening only and is not a medical diagnosis.</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    rows = "".join(
        f"<div class='sum-row'><span>{k}</span><b>{v}</b></div>"
        for k, v in r["inputs"].items()
    )
    st.markdown(
        f"<div class='sum-card'><div class='sum-title'>📋 Patient details used for this prediction</div>"
        f"<div class='sum-grid'>{rows}</div></div>",
        unsafe_allow_html=True
    )

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        if st.button("← New prediction", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()
