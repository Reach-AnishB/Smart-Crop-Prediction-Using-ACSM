from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

# NEW imports for dynamic graph
import matplotlib.pyplot as plt
import base64
from io import BytesIO

app = Flask(__name__)

# Load trained model bundle
obj = joblib.load("model.joblib")
model = obj["model"]
scaler = obj["scaler"]
le = obj["le"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    df = pd.read_csv(file)

    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])

    # Take first row
    sample = df.iloc[0].values.reshape(1, -1)
    sample_scaled = scaler.transform(sample)

    # Prediction
    pred = model.predict(sample_scaled)
    crop = le.inverse_transform(pred)[0]

    # -------- Dynamic confidence graph --------
    probs = model.predict_proba(sample_scaled)[0]
    labels = le.classes_

    # Top 5 crops
    top_idx = np.argsort(probs)[-5:][::-1]
    top_labels = labels[top_idx]
    top_probs = probs[top_idx]

    plt.figure(figsize=(6, 4))
    plt.bar(top_labels, top_probs)
    plt.ylabel("Probability")
    plt.title("Prediction Confidence (Top Crops)")
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    plt.close()
    buffer.seek(0)

    graph_base64 = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({
        "crop": crop,
        "confidence_graph": graph_base64
    })

if __name__ == "__main__":
    app.run(debug=True)
