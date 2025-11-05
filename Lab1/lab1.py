# =========================================================
# CYBR 520 - LAB 1 (Exploratory Data Analysis - Python Version)
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import os

# Create a folder for saving all plots
os.makedirs("plots", exist_ok=True)

# ---- Load Dataset ----
iris = datasets.load_iris(as_frame=True)
df = iris.frame
df.columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species"]
df["Species"] = df["Species"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

print(df.head())
print(df.info())

# ---- Descriptive Statistics ----
summary_stats = df.describe(include="all").T
print(summary_stats)

# Mean by Species
species_means = df.groupby("Species").mean(numeric_only=True)
print("\nMean by Species:\n", species_means)

# ---- Initial Visuals ----
sns.histplot(data=df, x="Sepal.Length", hue="Species", bins=20, kde=True, alpha=0.6)
plt.title("Distribution of Sepal Length by Species")
plt.savefig("plots/Initial_Sepal_Length_Distribution.png", dpi=300, bbox_inches="tight")
plt.show()

sns.boxplot(data=df, x="Species", y="Sepal.Width", palette="Set2")
plt.title("Sepal Width by Species")
plt.savefig("plots/Initial_Sepal_Width_Boxplot.png", dpi=300, bbox_inches="tight")
plt.show()

sns.scatterplot(data=df, x="Sepal.Length", y="Sepal.Width", hue="Species", style="Species")
plt.title("Sepal Dimensions by Species")
plt.savefig("plots/Initial_Sepal_Dimensions.png", dpi=300, bbox_inches="tight")
plt.show()

sns.scatterplot(data=df, x="Petal.Length", y="Petal.Width", hue="Species", style="Species")
plt.title("Petal Dimensions by Species")
plt.savefig("plots/Initial_Petal_Dimensions.png", dpi=300, bbox_inches="tight")
plt.show()

corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True)
plt.title("Feature Correlations (R-style Output)")
plt.savefig("plots/Initial_Feature_Correlations.png", dpi=300, bbox_inches="tight")
plt.show()

sns.pairplot(df, hue="Species", corner=True)
plt.suptitle("Pairwise Feature Relationships", y=1.02)
plt.savefig("plots/Initial_Pairwise_Feature_Relationships.png", dpi=300, bbox_inches="tight")
plt.show()

# ---- Grouped Stats ----
species_stats = df.groupby("Species").agg(["mean", "median", "std"])
species_stats.columns = ['_'.join(col).strip() for col in species_stats.columns.values]
print("Mean, Median, and Standard Deviation for Each Numeric Variable by Species:\n")
print(species_stats)
species_stats.to_csv("iris_species_stats.csv")
species_stats.to_html("iris_species_stats.html")


# =========================================================
# ---- Question 6: Histograms for Each Numeric Variable ----
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.histplot(df, x="Sepal.Length", bins=20, kde=True, color="skyblue", ax=axs[0, 0])
sns.histplot(df, x="Sepal.Width", bins=20, kde=True, color="lightgreen", ax=axs[0, 1])
sns.histplot(df, x="Petal.Length", bins=20, kde=True, color="salmon", ax=axs[1, 0])
sns.histplot(df, x="Petal.Width", bins=20, kde=True, color="gold", ax=axs[1, 1])
for ax in axs.flat:
    ax.set_xlabel("Centimeters")
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.suptitle("Q6: Histograms of Iris Dataset Features", fontsize=14, y=1.03)
plt.savefig("plots/Q6_Histograms.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# ---- Question 7: Boxplots by Species ----
# =========================================================
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
sns.boxplot(df, x="Species", y="Sepal.Length", palette="Set2", ax=axs[0, 0])
sns.boxplot(df, x="Species", y="Sepal.Width", palette="Set2", ax=axs[0, 1])
sns.boxplot(df, x="Species", y="Petal.Length", palette="Set2", ax=axs[1, 0])
sns.boxplot(df, x="Species", y="Petal.Width", palette="Set2", ax=axs[1, 1])
plt.tight_layout()
plt.suptitle("Q7: Boxplots of Iris Features by Species", fontsize=14, y=1.03)
plt.savefig("plots/Q7_Boxplots.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# ---- Question 8: Scatterplot Matrix ----
# =========================================================
sns.pairplot(df, hue="Species", corner=True, diag_kind="kde")
plt.suptitle("Q8: Scatterplot Matrix of Iris Dataset", y=1.02)
plt.savefig("plots/Q8_Scatterplot_Matrix.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# ---- Question 9: Correlation Heatmap ----
# =========================================================
corr = df.select_dtypes("number").corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True)
plt.title("Q9: Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/Q9_Correlation_Heatmap.png", dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# Question 10: Petal Dimensions by Species
# =========================================================
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="Petal.Length",
    y="Petal.Width",
    hue="Species",
    style="Species",
    s=70,                # equivalent to size=3 in ggplot
    alpha=0.8
)
plt.title("Petal Dimensions by Species")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.tight_layout()
plt.savefig("plots/Q10_Petal_Dimensions_by_Species.png", dpi=300, bbox_inches="tight")
plt.show()

# ---- Comment ----
# This plot shows that Setosa has distinctly smaller petals,
# while Versicolor and Virginica overlap slightly but differ in size ranges.


# =========================================================
# Question 11: Ratio of Petal Length to Petal Width by Species (Box Plot)
# =========================================================
# Compute the ratio
df["Petal.Ratio"] = df["Petal.Length"] / df["Petal.Width"]

plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="Species",
    y="Petal.Ratio",
    palette="Set3",
    showfliers=True,
    linewidth=1
)
# Add horizontal reference line like geom_hline(yintercept = 1)
plt.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
plt.title("Ratio of Petal Length to Petal Width by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length / Petal Width")
plt.tight_layout()
plt.savefig("plots/Q11_Ratio_Petal_Length_Width_by_Species.png", dpi=300, bbox_inches="tight")
plt.show()

# ---- Comment ----
# Petal ratios differ sharply among species; Setosa remains below ~1.5,
# while Virginica reaches much higher ratios.


# =========================================================
# Question 12: Correlation Analysis and Feature Relationships
# =========================================================
# Compute correlation matrix for numeric columns
corr = df.select_dtypes("number").corr()
print("\nCorrelation Matrix for Numeric Variables:\n", corr)

# Plot correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="coolwarm", square=True)
plt.title("Correlation Heatmap of Iris Features")
plt.tight_layout()
plt.savefig("plots/Q12_Correlation_Heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# Scatterplot of Sepal.Length vs Petal.Length colored by Species, sized by Petal.Width
plt.figure(figsize=(7, 5))
sns.scatterplot(
    data=df,
    x="Sepal.Length",
    y="Petal.Length",
    hue="Species",
    size="Petal.Width",
    sizes=(40, 200),
    alpha=0.7
)
plt.title("Iris Feature Relationships")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("plots/Q12_Iris_Feature_Relationships.png", dpi=300, bbox_inches="tight")
plt.show()

# ---- Comment ----
# The correlation matrix and scatterplot confirm that Petal.Length and Petal.Width
# are strongly correlated (r ≈ 0.96), and Sepal.Length increases with Petal.Length.
