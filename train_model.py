# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
import joblib
import json

# Load dataset
df = pd.read_csv("Reordered_Crop_Recommendation.csv")

# Separate features and label
X = df.drop(columns=["Crop"])
y = df["Crop"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the GBM model
model = make_pipeline(
    MinMaxScaler(),
    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model.pkl")

# Calculate and save ideal conditions
ideal_conditions = df.groupby("Crop").mean(numeric_only=True).round(2).to_dict(orient="index")
with open("ideal_conditions.json", "w") as f:
    json.dump(ideal_conditions, f, indent=2)

print("GBM model and ideal conditions saved.")
