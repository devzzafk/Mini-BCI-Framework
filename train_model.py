import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load features
df = pd.read_csv("../data/features.csv")

# For this beginner example, we'll create fake labels
# Let's say Ch1_mean + Ch2_mean > 0 → 'Focused', else 'Relaxed'
df['label'] = df['Ch1_mean'] + df['Ch2_mean']
df['label'] = df['label'].apply(lambda x: 1 if x > 0 else 0)  # 1 = Focused, 0 = Relaxed

# Features and labels
X = df.drop('label', axis=1)
y = df['label']

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)

# Check accuracy
print("Accuracy:", accuracy_score(y_test, pred))

# Save the model for later use
import joblib
joblib.dump(model, "../models/eeg_model.pkl")
print("Model trained and saved as eeg_model.pkl!")
