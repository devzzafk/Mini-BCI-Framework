import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Mini BCI Dashboard", layout="wide")
st.title(" Mini BCI Experiment Dashboard")
st.write("Simulate EEG signals, extract features, and predict mental state!")

# Function to simulate EEG
def simulate_eeg(n_channels=4, n_samples=250):
    data = np.random.randn(n_samples, n_channels)
    df = pd.DataFrame(data, columns=[f"Ch{i+1}" for i in range(n_channels)])
    return df

# Button to generate EEG
if st.button("Simulate EEG Session"):
    eeg_df = simulate_eeg()
    st.subheader("Simulated EEG Signals")
    st.line_chart(eeg_df)

    # Extract features
    features = {}
    for col in eeg_df.columns:
        features[f"{col}_mean"] = eeg_df[col].mean()
        features[f"{col}_var"] = eeg_df[col].var()
    features_df = pd.DataFrame([features])

    st.subheader("Extracted Features")
    st.dataframe(features_df)

    # Prediction
    try:
        model = joblib.load("eeg_model.pkl")
        pred = model.predict(features_df)[0]
        label = "Relaxed" if pred == 0 else "Focused"
    except:
        label = np.random.choice(["Relaxed", "Focused", "Neutral"])

    # Show gauge bar
    st.subheader("Predicted Mental State")
    if label == "✮⋆˙Relaxed":
        st.success(" ⋆˙⟡Relaxed")
    elif label == "Focused":
        st.info(" ꫂ❁Focused")
    else:
        st.warning(" Neutral")

    # Download CSV
    combined_df = pd.concat([eeg_df, features_df.reindex(eeg_df.index)], axis=1)
    csv = combined_df.to_csv(index=False)
    st.download_button("📥 Download EEG + Features CSV", csv, file_name="eeg_session.csv")
