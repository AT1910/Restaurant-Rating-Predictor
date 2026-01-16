from pathlib import Path

import streamlit as st
import numpy as np
import joblib

# --------------------------------------------------
# Streamlit config (MUST be first Streamlit command)
# --------------------------------------------------
st.set_page_config(
    page_title="Restaurant Rating Predictor",
    page_icon="🍴",
    layout="wide",
)

# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "Artifacts"

SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"
MODEL_PATH = ARTIFACTS_DIR / "restaurant_rating_predictor_model.pkl"

# --------------------------------------------------
# Safety checks (helps debugging on Streamlit Cloud)
# --------------------------------------------------
if not SCALER_PATH.exists():
    st.error(f"Scaler file not found: {SCALER_PATH}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
scaler = joblib.load(SCALER_PATH)
model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Restaurant Rating Predictor")
st.caption("Predict restaurant ratings using a machine learning model")
st.divider()

averagecost = st.number_input(
    "Estimated average cost for two",
    min_value=50,
    max_value=100000,
    value=1000,
    step=200,
)

tablebooking = st.selectbox(
    "Does the restaurant accept table bookings?",
    ("Yes", "No"),
)

onlinedelivery = st.selectbox(
    "Does the restaurant offer online delivery?",
    ("Yes", "No"),
)

pricerange = st.selectbox(
    "Price range (1 = Cheapest, 4 = Most Expensive)",
    (1, 2, 3, 4),
)

predictbutton = st.button("Predict Rating")
st.divider()

# --------------------------------------------------
# Feature engineering
# --------------------------------------------------
bookingstatus = 1 if tablebooking == "Yes" else 0
deliverystatus = 1 if onlinedelivery == "Yes" else 0

X_input = np.array([[averagecost, bookingstatus, deliverystatus, pricerange]])
X_scaled = scaler.transform(X_input)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if predictbutton:
    prediction = model.predict(X_scaled)[0]
    st.success(f"⭐ Predicted Restaurant Rating: **{prediction:.2f}**")
    # --------------------------------------------------
# Prediction
# --------------------------------------------------
if predictbutton:
    st.snow()
    prediction = model.predict(X_scaled)[0]
    st.success(f"⭐ Predicted Restaurant Rating: **{prediction:.2f}**")

