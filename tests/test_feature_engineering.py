import pytest
import pandas as pd
import numpy as np
from src.feature_engineering import (
    prepare_features,
    perform_pca,
    add_risk_flags_and_bins,
    add_interactions_and_polynomials,
    risk_scores
)


@pytest.fixture
def sample_data():
    """Fixture for creating a small sample dataframe for testing."""
    return pd.DataFrame({
        "age": [30, 55, 70],
        "trtbps": [120, 145, 160],
        "chol": [180, 250, 300],
        "thalachh": [150, 130, 90],
        "oldpeak": [1.0, 2.5, 3.0],
        "fbs": [0, 1, 0],
        "exng": [0, 1, 1],
        "output": [1, 0, 1]
    })


def test_add_risk_flags_and_bins(sample_data):
    df = add_risk_flags_and_bins(sample_data.copy())

    # Check new columns exist
    expected_cols = [
        "is_hypertensive", "is_hyperchol", "is_oldpeak_high",
        "age_group", "chol_group", "bp_group", "thalachh_group"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Check categorical binning
    assert df["age_group"].iloc[0] == "young"
    assert df["chol_group"].iloc[1] == "high"
    assert df["bp_group"].iloc[2] == "hypertension"


def test_add_interactions_and_polynomials(sample_data):
    df = add_interactions_and_polynomials(sample_data.copy())

    expected_new_cols = [
        "age_chol_ratio", "chol_squared", "log_chol",
        "bp_chol_product", "bp_oldpeak_interaction"
    ]
    for col in expected_new_cols:
        assert col in df.columns

    # No NaN values introduced
    assert df.isnull().sum().sum() == 0


def test_risk_scores(sample_data):
    df = risk_scores(sample_data.copy())

    # Check expected columns
    assert "risk_count" in df.columns
    assert "weighted_risk_score" in df.columns
    assert "bp_chol_risk" in df.columns

    # Risk count logic check
    assert df["risk_count"].iloc[1] > df["risk_count"].iloc[0]


def test_prepare_features(sample_data):
    df = prepare_features(sample_data.copy())

    # Ensure combined transformations created columns
    assert "weighted_risk_score" in df.columns
    assert "chol_squared" in df.columns
    assert "is_hypertensive" in df.columns


def test_perform_pca(sample_data):
    X = sample_data.drop(columns=["output"])
    X_train, X_test = X.iloc[:2], X.iloc[2:].copy()

    X_train_pca, X_test_pca, pca = perform_pca(X_train, X_test)

    # PCA should reduce dimensions
    assert X_train_pca.shape[1] <= X_train.shape[1]

    # PCA output should be numeric and not empty
    assert np.issubdtype(X_train_pca.dtypes[0], np.number)

    # Explained variance ratio must exist
    assert hasattr(pca, "explained_variance_ratio_")
