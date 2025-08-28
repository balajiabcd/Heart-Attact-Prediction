
import os
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing


from src import visualize as viz

# Ensure test save path exists
SAVE_PATH = "static/data_analysis"
os.makedirs(SAVE_PATH, exist_ok=True)


@pytest.fixture
def sample_df():
    """Fixture for a sample dataframe mimicking heart dataset."""
    return pd.DataFrame({
        "age": [29, 45, 60, 50, 39],
        "trtbps": [120, 140, 150, 130, 125],
        "chol": [200, 240, 180, 220, 210],
        "thalachh": [150, 160, 140, 155, 145],
        "oldpeak": [1.0, 2.3, 1.5, 0.8, 1.2],
        "output": [1, 0, 1, 0, 1]
    })


def test_plot_class_distribution(sample_df):
    file_path = f"{SAVE_PATH}/class_distribution.png"
    viz.plot_class_distribution(sample_df, target_col="output", save_path=file_path)
    assert os.path.exists(file_path)


def test_plot_correlation(sample_df):
    file_path = f"{SAVE_PATH}/correlation_heatmap_raw.png"
    viz.plot_correlation(sample_df, save_path=file_path)
    assert os.path.exists(file_path)


def test_plot_feature_boxplots(sample_df):
    numeric_features = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
    viz.plot_feature_boxplots(sample_df, numeric_features, target_col="output")
    for col in numeric_features:
        assert os.path.exists(f"{SAVE_PATH}/boxplot_{col}.png")


def test_plot_histograms(sample_df):
    numeric_features = ["age", "trtbps", "chol", "thalachh", "oldpeak"]
    viz.plot_histograms(sample_df, numeric_features)
    for col in numeric_features:
        assert os.path.exists(f"{SAVE_PATH}/hist_{col}.png")


def test_plot_scatter(sample_df):
    file_path = f"{SAVE_PATH}/scatter_age_thalachh_raw.png"
    viz.plot_scatter(sample_df, "age", "thalachh", target_col="output", save_path=file_path)
    assert os.path.exists(file_path)


def test_plot_model_performance():
    results_df = pd.DataFrame({
        "model": ["m1", "m2"],
        "Accuracy": [0.8, 0.9],
        "Precision": [0.75, 0.88],
        "Recall": [0.70, 0.85],
        "F1 Score": [0.72, 0.86],
    })
    file_path = f"{SAVE_PATH}/model_performance.png"
    viz.plot_model_performance(results_df)
    assert os.path.exists(file_path)
