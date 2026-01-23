import pandas as pd

# Load the filtered EEG data
df = pd.read_csv("../data/simulated_eeg_filtered.csv")

# Create a new DataFrame for features
features = pd.DataFrame()

# For each channel, calculate simple features: mean & variance
for col in df.columns:
    features[f"{col}_mean"] = [df[col].mean()]
    features[f"{col}_var"] = [df[col].var()]

# Save the features to CSV
features.to_csv("../data/features.csv", index=False)
print("Features extracted and saved as features.csv!")
print(features)
