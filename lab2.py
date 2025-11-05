# =========================================================
# CYBR 520 - LAB 2 (Statistical Classification and Decision Trees)
# =========================================================

import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score, roc_curve
)

# ---------------------------------------------------------
# 1–3. Load and Inspect Dataset
# ---------------------------------------------------------
print("Current working directory:", os.getcwd())
df = pd.read_csv("spambase.csv")

# Clean target labels
df["type"] = df["type"].str.lower().str.strip()
df["type"] = df["type"].replace({"nonspam": 0, "spam": 1})

print("Shape of dataset:", df.shape)
print("Number of observations:", df.shape[0])
print("Number of attributes:", df.shape[1])

# ---------------------------------------------------------
# 4–5. Split Features and Target
# ---------------------------------------------------------
X = df.drop(columns="type")
y = df["type"]

# 70/30 train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
print(f"Training set: {X_train.shape[0]} rows")
print(f"Testing set: {X_test.shape[0]} rows")

# ---------------------------------------------------------
# 6–7. Build and Train Decision Tree
# ---------------------------------------------------------
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# ---------------------------------------------------------
# 8. Visualize Decision Tree
# ---------------------------------------------------------
os.makedirs("plots", exist_ok=True)
plt.figure(figsize=(25, 12))
plot_tree(
    tree,
    feature_names=list(X.columns),
    class_names=["Nonspam", "Spam"],
    filled=True,
    rounded=True,
    fontsize=10,
    max_depth=3
)
plt.title("Decision Tree - Spam vs. Nonspam", fontsize=18, weight="bold")
plt.tight_layout()
plt.savefig("plots/Decision_Tree_Spam_vs_Nonspam.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 9. Feature Importance Visualization
# ---------------------------------------------------------
importance = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Important Features:")
print(importance.head(10))

plt.figure(figsize=(8, 6))
sns.barplot(x=importance.head(10), y=importance.head(10).index, palette="viridis")
plt.title("Top 10 Feature Importances - Spam Classifier")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("plots/Feature_Importance.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 10–11. Predictions and Probabilities
# ---------------------------------------------------------
y_pred = tree.predict(X_test)
y_prob = tree.predict_proba(X_test)[:, 1]

predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred, "Prob_Spam": y_prob})
print(predictions.head())

# ---------------------------------------------------------
# 12. Confusion Matrix Visualization
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
labels = np.array([["True Negative", "False Positive"], ["False Negative", "True Positive"]])
cm_percent = cm / cm.sum() * 100
annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i, j] = f"{labels[i, j]}\n({cm_percent[i, j]:.1f}%)"

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=annot, fmt="", cmap="Reds",
            xticklabels=["Nonspam", "Spam"], yticklabels=["Nonspam", "Spam"],
            cbar_kws={"label": "Count"})
plt.title("Confusion Matrix - Decision Tree Spam Classifier", fontsize=14, weight="bold")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.savefig("plots/Confusion_Matrix.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# 13–16. Model Performance Metrics
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")

print("\nAdditional Metrics:")
print(f"Matthews Corrcoef: {matthews_corrcoef(y_test, y_pred):.4f}")
print(f"Cohen Kappa:       {cohen_kappa_score(y_test, y_pred):.4f}")
print(f"ROC AUC:           {roc_auc_score(y_test, y_prob):.4f}")
