
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os



def ensure_dir(path="static/data_analysis"):
    os.makedirs(path, exist_ok=True)
    return path

SAVE_PATH = "static/data_analysis"
os.makedirs(SAVE_PATH, exist_ok=True)


def plots_data_analysis(df):
    """Plots for raw data analysis just after importing data."""
    plot_class_distribution(df, target_col="output", save_path=f"{SAVE_PATH}/class_distribution.png")
    plot_correlation(df, save_path=f"{SAVE_PATH}/correlation_heatmap_raw.png")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [f for f in numeric_features if f != "output"]
    plot_histograms(df, numeric_features)
    plot_feature_boxplots(df, numeric_features, target_col="output")
    # Example scatterplots for raw data
    scatter_pairs = [("age","thalachh"),("trtbps","chol"),("oldpeak","age")]
    for x, y in scatter_pairs:
        if x in df.columns and y in df.columns:
            plot_scatter(df, x, y, target_col="output", save_path=f"{SAVE_PATH}/scatter_{x}_{y}_raw.png")



def plots_all_features(df, results_df=None):
    """Plots for feature-engineered data after preprocessing and feature engineering."""
    plot_class_distribution(df, target_col="output", save_path=f"{SAVE_PATH}/class_distribution_fe.png")
    plot_correlation(df, save_path=f"{SAVE_PATH}/correlation_heatmap_fe.png")
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_features = [f for f in numeric_features if f != "output"]
    plot_histograms(df, numeric_features)
    plot_feature_boxplots(df, numeric_features, target_col="output")
    # Scatterplots for feature-engineered data
    scatter_pairs = [("age","thalachh"),("trtbps","chol"),("oldpeak","age"),("risk_count","weighted_risk_score")]
    for x, y in scatter_pairs:
        if x in df.columns and y in df.columns:
            plot_scatter(df, x, y, target_col="output", save_path=f"{SAVE_PATH}/scatter_{x}_{y}_fe.png")
    # If model results available, plot performance
    if results_df is not None:
        plot_model_performance(results_df)





def plot_class_distribution(df, target_col="output", save_path=None):
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x=target_col, palette="Set2")
    plt.title(f"{target_col} Distribution")
    plt.xlabel(target_col)
    plt.ylabel("Count")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()



def plot_correlation(df, save_path=None):
    plt.figure(figsize=(12,10))
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()



def plot_feature_boxplots(df, numeric_features, target_col="output"):
    for col in numeric_features:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, x=target_col, y=col, palette="Set3")
        plt.title(f"{col} vs {target_col}")
        plt.xlabel(target_col)
        plt.ylabel(col)
        plt.savefig(f"{SAVE_PATH}/boxplot_{col}.png", bbox_inches="tight")
        plt.close()



def plot_histograms(df, numeric_features):
    for col in numeric_features:
        plt.figure(figsize=(6,4))
        sns.histplot(df[col], bins=30, kde=True, color="skyblue")
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.savefig(f"{SAVE_PATH}/hist_{col}.png", bbox_inches="tight")
        plt.close()



def plot_scatter(df, x, y, target_col="output", save_path=None):
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x=x, y=y, hue=target_col, palette="Set1")
    plt.title(f"{y} vs {x}")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()



def plot_model_performance(results_df):
    plt.figure(figsize=(10,6))
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
    results_df.set_index("model")[metrics].plot(kind="bar", figsize=(12,6))
    plt.title("ML Model Performance Comparison")
    plt.ylabel("Score")
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.ylim(0,1)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}/model_performance.png")
    plt.close()




