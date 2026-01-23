import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("../models/eeg_model.pkl")

st.title("🧠 Mini BCI Dashboard")
st.write("Simulate EEG and see real-time mental state predictions.")

# Button to simulate EEG
if st.button("Simulate EEG Session"):
    fs = 250
    seconds = 5
    channels = 4
    time_points = np.arange(0, seconds, 1/fs)
    
    # Simulate EEG
    new_data = np.random.randn(len(time_points), channels)
    df_new = pd.DataFrame(new_data, columns=[f'Ch{i+1}' for i in range(channels)])
    
    # Extract features
    features = {}
    for col in df_new.columns:
        features[f"{col}_mean"] = df_new[col].mean()
        features[f"{col}_var"] = df_new[col].var()
    features_df = pd.DataFrame([features])
    
    # Predict
    prediction = model.predict(features_df)[0]
    label = "Relaxed" if prediction == 0 else "Focused"
    
    st.subheader(f"Prediction: {label}")
    
    # Plot EEG signals
    st.line_chart(df_new)
