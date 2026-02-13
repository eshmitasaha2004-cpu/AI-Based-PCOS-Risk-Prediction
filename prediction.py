# ===============================
# PCOS Prediction ML Project
# Using 50,000 Synthetic Dataset
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1️⃣ Load Dataset
df = pd.read_csv("pcos_50000_dataset.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# 2️⃣ Separate Features and Target
X = df.drop("PCOS", axis=1)
y = df["PCOS"]

# 3️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Logistic Regression Model
# ===============================

log_model = LogisticRegression(max_iter=2000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

# ===============================
# Random Forest Model
# ===============================

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ===============================
# Feature Importance (Random Forest)
# ===============================

importances = rf_model.feature_importances_
features = X.columns

feature_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(feature_df)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=feature_df)
plt.title("Feature Importance for PCOS Prediction")
plt.show()
