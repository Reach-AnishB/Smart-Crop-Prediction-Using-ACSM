import joblib
import numpy as np

# Load the saved model bundle
obj = joblib.load("model.joblib")
model = obj["model"]
scaler = obj["scaler"]
le = obj["le"]

# Example input values (you can change these)
sample = np.array([[90, 42, 43, 20.8, 82, 6.5, 200]])
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
crop = le.inverse_transform(prediction)[0]

print("Recommended Crop:", crop)
