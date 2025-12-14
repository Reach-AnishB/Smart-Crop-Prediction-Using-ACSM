import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("crop.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Load trained objects
obj = joblib.load("model.joblib")
model = obj["model"]
scaler = obj["scaler"]
le = obj["le"]

# Encode labels
y_enc = le.transform(y)

# Scale features (IMPORTANT)
X_scaled = scaler.transform(X)

# Define scoring metrics
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, average="weighted", zero_division=0),
    "recall": make_scorer(recall_score, average="weighted", zero_division=0),
    "f1": make_scorer(f1_score, average="weighted", zero_division=0)
}

# 5-Fold Cross Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_validate(model, X_scaled, y_enc, cv=cv, scoring=scoring)

# Mean scores
accuracy = scores["test_accuracy"].mean() * 100
precision = scores["test_precision"].mean() * 100
recall = scores["test_recall"].mean() * 100
f1 = scores["test_f1"].mean() * 100

# Standard deviation (variation)
accuracy_std = scores["test_accuracy"].std() * 100
precision_std = scores["test_precision"].std() * 100
recall_std = scores["test_recall"].std() * 100
f1_std = scores["test_f1"].std() * 100

# Plot data
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
values = [accuracy, precision, recall, f1]
errors = [accuracy_std, precision_std, recall_std, f1_std]
colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

plt.figure(figsize=(8, 4))
bars = plt.bar(
    metrics,
    values,
    yerr=errors,
    capsize=6,
    color=colors,
    edgecolor="black"
)

plt.ylim(95, 100)  # Zoomed for clarity
plt.ylabel("Score (%)")
plt.title("Performance Metrics (5-Fold Cross Validation)")
plt.grid(axis="y", linestyle="--", alpha=0.4)

# Value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.15,
        f"{height:.2f}%",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("static/performance_metrics.png")
plt.close()

print("Updated performance_metrics.png generated successfully")
