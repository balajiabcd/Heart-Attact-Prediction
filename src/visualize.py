import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_class_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(x="diagnosis", data=df)
    plt.title("Class Distribution")
    plt.show()
