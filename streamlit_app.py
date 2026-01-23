import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Mini BCI Experiment Dashboard 🧠")

st.write("Simulate EEG, extract features, and predict mental states.")

# Button to simulate EEG
if st.button("Simulate EEG & Predict"):
    # Step 1: Simulate EEG data
    fs = 250
    seconds = 5
    channels = 4
    time_points = np.arange(0, seconds, 1/fs)
    new_data = np.random.randn(len(time_points), channels)
    df_new = pd.DataFrame(new_data, columns=[f"Ch{i+1}" for i in range(channels)])
    
    st.subheader("Simulated EEG Signals")
    st.line_chart(df_new)

    # Step 2: Extract simple features
    features = {}
    for col in df_new.columns:
        features[f"{col}_mean"] = df_new[col].mean()
        features[f"{col}_var"] = df_new[col].var()
    
    features_df = pd.DataFrame([features])
    st.subheader("Extracted Features")
    st.dataframe(features_df)

    # Step 3: Load the trained model and predict
    try:
        model = joblib.load("eeg_model.pkl")
        prediction_numeric = model.predict(features_df)[0]
        label = "Relaxed" if prediction_numeric == 0 else "Focused"
        st.success(f"Predicted Mental State: **{label}**")
    except Exception as e:
        st.error("Error loading model, showing random result")
        random_label = np.random.choice(["Relaxed", "Focused"])
        st.info(f"Predicted Mental State (simulated): **{random_label}**")
