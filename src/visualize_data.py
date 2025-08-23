# scripts/visualize_data.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

# Load dataset
df = pd.read_csv("data/breast-cancer.csv")

# Encode target for plotting
df['diagnosis_encoded'] = df['diagnosis'].map({'M': 1, 'B': 0})


# 1. Class distribution
plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis', data=df)
plt.title("Class Distribution")
plt.savefig("images/class_distribution.png", bbox_inches='tight')
plt.close()

# 2. Correlation heatmap (drop 'id' column)
plt.figure(figsize=(14,12))
sns.heatmap(df.drop(columns=['id','diagnosis']).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.savefig("images/correlation_heatmap.png", bbox_inches='tight')
plt.close()

# 3. Boxplots for key numeric features
key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']
for feature in key_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='diagnosis', y=feature, data=df)
    plt.title(f"{feature} vs Diagnosis")
    plt.savefig(f"images/{feature}_boxplot.png", bbox_inches='tight')
    plt.close()

# 4. Scatter plot: concavity_mean vs concave points_mean
plt.figure(figsize=(6,4))
sns.scatterplot(x='concavity_mean', y='concave points_mean', hue='diagnosis', data=df)
plt.title("Concavity Mean vs Concave Points Mean")
plt.savefig("images/concavity_scatter.png", bbox_inches='tight')
plt.close()

# 5. Pairplot for selected features
selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'diagnosis']
sns.pairplot(df[selected_features], hue='diagnosis')
plt.savefig("images/pairplot_selected_features.png", bbox_inches='tight')
plt.close()

print("All visualizations have been saved to the 'images/' folder.")
