# plot_feature_importance.py
import joblib
import pandas as pd
import matplotlib.pyplot as plt

obj = joblib.load("model.joblib")
model = obj["model"]
feat_names = ['N','P','K','temperature','humidity','ph','rainfall']
importances = model.feature_importances_

df = pd.DataFrame({'feature': feat_names, 'importance': importances})
df = df.sort_values(by='importance', ascending=True)

plt.figure(figsize=(8,4))
plt.barh(df['feature'], df['importance'])
plt.xlabel('Importance')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
