import joblib
import pandas as pd

# Load the saved model
obj = joblib.load("model.joblib")
model = obj["model"]

# Feature names in order
feat_names = ['N','P','K','temperature','humidity','ph','rainfall']

# Extract feature importances from RandomForest
importances = model.feature_importances_

# Create a DataFrame
df = pd.DataFrame({
    'feature': feat_names,
    'importance': importances
}).sort_values(by='importance', ascending=False).reset_index(drop=True)

# Print the feature importance table
print(df.to_string(index=False))

# Save the table to a CSV file
df.to_csv("feature_importance.csv", index=False)
print("\nSaved feature_importance.csv")
