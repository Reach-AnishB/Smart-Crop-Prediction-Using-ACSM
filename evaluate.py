# evaluate.py
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

MODEL_FILE = "model.joblib"
DATA = "data/crop.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# load model
obj = joblib.load(MODEL_FILE)
clf = obj['model']
scaler = obj['scaler']
le = obj['le']

# load data and prepare test split the same way you trained
df = pd.read_csv(DATA)
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']
y_enc = le.transform(y)  # use same encoder

# split (must match training random_state and stratify)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=0, stratify=y_enc)

X_test_s = scaler.transform(X_test)
y_pred = clf.predict(X_test_s)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

print(f"Accuracy: {acc:.4f}\n")
print(report)
cm_df.to_csv(os.path.join(OUT_DIR, "confusion_matrix.csv"))
with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report)
print("Saved metrics to models/")
