import numpy as np
import pandas as pd

fs = 250  # samples per second
seconds = 5
channels = 4
samples = 10  # number of simulated sessions

all_features = []

for s in range(samples):
    time = np.arange(0, seconds, 1/fs)
    data = np.random.randn(len(time), channels)
    df = pd.DataFrame(data, columns=[f'Ch{i+1}' for i in range(channels)])
    
    # Calculate mean & variance for each channel (features)
    features = {}
    for col in df.columns:
        features[f"{col}_mean"] = df[col].mean()
        features[f"{col}_var"] = df[col].var()
    all_features.append(features)

# Save all features as CSV
features_df = pd.DataFrame(all_features)
features_df.to_csv("../data/features.csv", index=False)
print("Multiple features extracted and saved as features.csv!")
print(features_df)
