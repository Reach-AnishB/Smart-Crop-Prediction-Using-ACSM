# mahalanobis_ranking.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import pinv
import joblib

DATA_PATH = os.path.join("data", "crop.csv")
OUT_IMAGE = "mahalanobis_ranking.png"

# --- configuration: edit sample values here (N,P,K,temp,humidity,ph,rainfall) ---
sample = np.array([90, 42, 43, 20.8, 82.0, 6.5, 200.0])

# --- load dataset ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Put your CSV at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# ensure columns exist
cols = ['N','P','K','temperature','humidity','ph','rainfall']
for c in cols:
    if c not in df.columns:
        raise KeyError(f"Column '{c}' not found in dataset")

X = df[cols].values
y = df['label'].values

# compute global covariance (with rows = examples)
# use pseudo-inverse (pinv) to avoid singular matrix errors
cov = np.cov(X, rowvar=False)
cov_inv = pinv(cov)

# compute class-wise means and Mahalanobis distance
classes = np.unique(y)
distances = []
for cls in classes:
    cls_X = X[y == cls]
    mean_vec = cls_X.mean(axis=0)
    diff = sample - mean_vec
    # Mahalanobis distance (no sqrt needed for ranking, but we use sqrt to match intuitive scale)
    md = np.sqrt(float(diff.T.dot(cov_inv).dot(diff)))
    distances.append((cls, md))

# sort ascending (lower distance = better)
distances = sorted(distances, key=lambda x: x[1])

# prepare for plotting (top N crops)
top = distances  # show all; change to distances[:5] for top-5
crops = [t[0] for t in top]
vals = [t[1] for t in top]

# Plot
plt.figure(figsize=(10,5))
bars = plt.bar(crops, vals)
plt.title("Top Crop Recommendations by Mahalanobis Distance")
plt.ylabel("Mahalanobis Distance (lower is better)")
plt.xlabel("Crop")
plt.xticks(rotation=25, ha='right')
plt.ylim(0, max(vals)*1.2)

# annotate values on bars
for bar, v in zip(bars, vals):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + max(vals)*0.02, f"{v:.2f}",
             ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_IMAGE)
plt.show()

print(f"Saved ranking image to: {OUT_IMAGE}")
print("Top recommendation (lowest distance):", crops[0], f"({vals[0]:.2f})")

