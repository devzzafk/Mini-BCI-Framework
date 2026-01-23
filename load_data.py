import pandas as pd
import matplotlib.pyplot as plt

# Load the simulated EEG
df = pd.read_csv("../data/simulated_eeg.csv")
print(df.head())  # Shows first 5 rows

# Plot the first channel
plt.plot(df['Ch1'])
plt.title("Simulated EEG - Channel 1")
plt.xlabel("Time points")
plt.ylabel("Signal")
plt.show()
