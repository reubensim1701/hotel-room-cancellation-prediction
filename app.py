# app.py
# Hotel Booking Cancellation Predictor (Streamlit) - Corrected Version
# This version is simplified to match the features used in the trained model.
# It removes unused input fields like 'agent', 'company', 'lead_time', etc.

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(page_title="Hotel Cancellation Predictor", page_icon="🏨", layout="centered")
st.markdown("## 🏨 Hotel Booking Cancellation Predictor")
st.caption("Fill the booking details on the left. The app loads cancellation_model.pkl and predicts cancellation likelihood.")

# ---------------------------------
# Constants
# ---------------------------------
# These are the top countries from the notebook, used for the 'country_grouped' feature.
TOP_COUNTRIES = ["PRT", "GBR", "FRA", "ESP", "DEU", "USA", "BRA", "ITA", "NLD", "IRL"]
COUNTRY_GROUP_OPTIONS = TOP_COUNTRIES + ["Other"]

def auto_country_group(code: str) -> str:
    """Groups a country code into 'Other' if it's not in the top countries list."""
    code_up = (code or "").strip().upper()
    return code_up if code_up in TOP_COUNTRIES else "Other"

# ---------------------------------
# Model loader
# ---------------------------------
@st.cache_resource(show_spinner=True)
def load_model(model_path: str = "cancellation_model.pkl"):
    """Loads the pickled model file."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at: {model_file.resolve()}")
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    try_force_xgb_cpu(model)
    return model

def try_force_xgb_cpu(model):
    """Attempts to set XGBoost to run on CPU to avoid GPU/device mismatch warnings."""
    try:
        params = model.get_params(deep=True)
    except Exception:
        return
    updates = {}
    for k in params.keys():
        if k.endswith("__device"):
            updates[k] = "cpu"
        if k.endswith("__predictor"):
            updates[k] = "cpu_predictor"
        if k.endswith("__tree_method"):
            updates[k] = "hist"
    if updates:
        try:
            model.set_params(**updates)
        except Exception:
            pass

# ---------------------------------
# Sidebar UI
# ---------------------------------
with st.sidebar:
    st.header("Booking details")

    # Load the model
    try:
        model = load_model("cancellation_model.pkl")
    except Exception as e:
        st.error("Could not load the model 'cancellation_model.pkl'.")
        st.exception(e)
        st.stop()

    # --- Input fields for the features the model was trained on ---

    st.markdown("### Booking & Channel")
    market_segment = st.selectbox("Market segment", ["Direct", "Corporate", "Online TA", "Offline TA/TO", "Complementary", "Groups", "Aviation", "Undefined"])
    customer_type = st.selectbox("Customer type", ["Transient", "Contract", "Transient-Party", "Group"])
    deposit_type = st.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])

    st.markdown("### Room")
    reserved_room_type = st.selectbox("Reserved room type", list("ABCDEFGHLP"))

    st.markdown("### History & Status")
    previous_cancellations = st.slider("Previous cancellations", 0, 30, 0)
    booking_changes = st.slider("Booking changes", 0, 50, 0)
    days_in_waiting_list = st.slider("Days in waiting list", 0, 400, 0)

    st.markdown("### Requests")
    required_car_parking_spaces = st.slider("Required car parking spaces", 0, 10, 0)
    total_of_special_requests = st.slider("Total special requests", 0, 10, 0)

    st.markdown("### Country")
    # This input is used to generate the 'country_grouped' feature
    country_value = st.selectbox("Country (ISO 3-letter)", TOP_COUNTRIES, index=0)
    country_grouped_value = auto_country_group(country_value)
    st.info(f"This will be categorized as **'{country_grouped_value}'** for the model.")


    st.markdown("### Prediction Settings")
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01, help="Probability above which a booking is flagged as a likely cancellation.")

    # Predict button
    predict_clicked = st.button("Predict Cancellation")

# ---------------------------------
# Main Panel - Prediction and Output
# ---------------------------------

def build_input_df() -> pd.DataFrame:
    """Creates a DataFrame from the sidebar inputs in the correct format for the model."""
    input_row = {
        # Numerical Features
        "previous_cancellations": int(previous_cancellations),
        "booking_changes": int(booking_changes),
        "days_in_waiting_list": int(days_in_waiting_list),
        "required_car_parking_spaces": int(required_car_parking_spaces),
        "total_of_special_requests": int(total_of_special_requests),
        # Categorical Features
        "market_segment": market_segment,
        "deposit_type": deposit_type,
        "customer_type": customer_type,
        "reserved_room_type": reserved_room_type,
        "country_grouped": country_grouped_value,
    }
    # The model expects the columns in a specific order. We define it here to be safe.
    expected_columns = [
        'market_segment', 'previous_cancellations', 'booking_changes',
        'deposit_type', 'days_in_waiting_list', 'customer_type',
        'reserved_room_type', 'required_car_parking_spaces',
        'total_of_special_requests', 'country_grouped'
    ]
    return pd.DataFrame([input_row])[expected_columns]

def predict_with_model(mdl, df: pd.DataFrame):
    """Makes predictions using the loaded model."""
    # The model pipeline handles preprocessing, so we pass the raw dataframe
    label = mdl.predict(df)[0]
    prob = None
    if hasattr(mdl, "predict_proba"):
        probs = mdl.predict_proba(df)
        # Get the probability of the positive class (1, for 'canceled')
        prob = float(probs[0, 1])
    return int(label), prob, df

if predict_clicked:
    try:
        X_input = build_input_df()
        with st.spinner("Predicting..."):
            pred_label, prob, X_used = predict_with_model(model, X_input)

        st.markdown("---")
        st.markdown("### 📊 Prediction Result")

        if prob is not None:
            will_cancel = prob >= threshold
            pct = int(round(prob * 100))
            
            # Display metric and progress bar
            st.metric("Cancellation Probability", f"{pct}%", f"Threshold: {int(threshold*100)}%")
            st.progress(min(max(prob, 0.0), 1.0))
            
            # Display final verdict
            if will_cancel:
                st.error("🔴 High likelihood of cancellation")
            else:
                st.success("🟢 Likely to stay (not canceled)")
        else:
            # Fallback if predict_proba is not available
            will_cancel = bool(pred_label == 1)
            st.info("Model does not provide probabilities; showing class prediction only.")
            if will_cancel:
                st.error("🔴 Predicted: Cancellation")
            else:
                st.success("🟢 Predicted: Not Canceled")

        # Expander to show the features sent to the model
        with st.expander("See input features used for prediction"):
            # Transpose the dataframe for better readability
            st.dataframe(X_used.T.rename(columns={0: 'Value'}), use_container_width=True)

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.caption("Tip: Ensure the sidebar fields match the features your model expects.")
else:
    st.info("Fill out the booking details in the sidebar and click 'Predict Cancellation'.")
