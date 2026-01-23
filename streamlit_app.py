import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Import your modules if you have them, else we simulate inline
# from preprocessing import filter_data
# from experiments import predict_state

st.title("Mini BCI Experiment Framework 🧠")

st.write("Simulate EEG signals, extract features, and predict mental state.")

# -----------------------------
# Step 1: Simulate EEG signals
# -----------------------------
if st.button("Simulate EEG"):
    # Simulate 4-channel EEG data (10 samples)
    eeg_data = pd.DataFrame({
        "Ch1": np.random.randn(10),
        "Ch2": np.random.randn(10),
        "Ch3": np.random.randn(10),
        "Ch4": np.random.randn(10)
    })
    st.write("Simulated EEG Data:")
    st.dataframe(eeg_data)

    # -----------------------------
    # Step 2: Extract simple features
    # -----------------------------
    features = pd.DataFrame({
        "Ch1_mean": [eeg_data["Ch1"].mean()],
        "Ch1_var": [eeg_data["Ch1"].var()],
        "Ch2_mean": [eeg_data["Ch2"].mean()],
        "Ch2_var": [eeg_data["Ch2"].var()],
        "Ch3_mean": [eeg_data["Ch3"].mean()],
        "Ch3_var": [eeg_data["Ch3"].var()],
        "Ch4_mean": [eeg_data["Ch4"].mean()],
        "Ch4_var": [eeg_data["Ch4"].var()],
    })
    st.write("Extracted Features:")
    st.dataframe(features)

    # -----------------------------
    # Step 3: Predict Mental State
    # -----------------------------
    # Try to load your trained model
    try:
        model = joblib.load("experiments/eeg_model.pkl")
        prediction = model.predict(features)[0]
        st.success(f"Predicted Mental State: {prediction}")
    except:
        st.warning("No trained model found. Using simulated prediction.")
        prediction = np.random.choice(["Relaxed", "Focused", "Neutral"])
        st.info(f"Predicted Mental State: {prediction}")
