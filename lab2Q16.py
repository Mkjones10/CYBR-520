"""
CYBR 520 – Lab 2
Question 16: Independent Classification – Cybersecurity Attacks Dataset
Author(s): LaShondra Edwards, Maxine Jones, Sean Plaisted, Sophia Walker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# === Load Dataset ===
df = pd.read_csv("cybersecurity_attacks.csv")

label_col = "Attack Type"

# === Drop Identifiers ===
drop_cols = [
    "Source IP Address", "Destination IP Address", "Payload Data",
    "User Information", "Device Information", "Geo-location Data",
    "Proxy Information", "Malware Indicators", "Alerts/Warnings",
    "Firewall Logs", "IDS/IPS Alerts", "Attack Signature"
]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# === Time-based Feature ===
if "Timestamp" in df.columns:
    df["HourOfDay"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.hour.fillna(0).astype(int)
    df = df.drop(columns=["Timestamp"])

# === Feature Lists ===
numeric_features = ["Source Port", "Destination Port", "Packet Length", "Anomaly Scores", "HourOfDay"]
categorical_features = [
    "Protocol", "Packet Type", "Traffic Type", "Action Taken",
    "Severity Level", "Network Segment", "Log Source"
]

num_feats = [c for c in numeric_features if c in df.columns]
cat_feats = [c for c in categorical_features if c in df.columns]

# === Prepare Data ===
X_raw = df[num_feats + cat_feats].copy()
y_raw = df[label_col]

le = LabelEncoder()
y = le.fit_transform(y_raw)
class_names = le.classes_

X = pd.get_dummies(X_raw, columns=cat_feats, drop_first=False).fillna(0)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)

# === Train Model ===
clf = DecisionTreeClassifier(min_samples_split=4, random_state=7)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cv   = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()

print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1 Score:  {f1:.3f}")
print(f"CV (5-fold): {cv:.3f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure()
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.title("Confusion Matrix (Decision Tree)")
plt.xlabel("Predicted")
plt.ylabel("True")
for (i, j), z in np.ndenumerate(cm):
    plt.text(j, i, str(z), ha="center", va="center", color="black")
plt.tight_layout()
plt.savefig("plots/confusion_matrix2.png", dpi=300)

# === Tree Visualization ===
plt.figure(figsize=(12, 6))
plot_tree(clf, max_depth=3, feature_names=X.columns, class_names=class_names, filled=True)
plt.title("Decision Tree (Top 3 Levels)")
plt.tight_layout()
plt.savefig("plots/decision_tree_top3.png", dpi=300)

print("Saved: plots/confusion_matrix2.png and plots/decision_tree_top3.png")
