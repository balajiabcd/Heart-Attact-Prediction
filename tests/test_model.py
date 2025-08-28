import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from src.model import train_and_evaluate, run_ml_models


@pytest.fixture
def sample_data():
    """Generate a small classification dataset."""
    X, y = make_classification(
        n_samples=50, n_features=5, n_informative=3,
        n_redundant=0, random_state=42
    )
    X_train, X_test = X[:30], X[30:]
    y_train, y_test = y[:30], y[30:]
    return X_train, X_test, y_train, y_test


def test_train_and_evaluate_returns_metrics(sample_data, tmp_path):
    X_train, X_test, y_train, y_test = sample_data

    model = LogisticRegression(max_iter=200, random_state=42)
    save_path = tmp_path / "models"
    save_path.mkdir()

    # Patch joblib.dump to prevent actual file creation in the wrong path
    with patch("src.model.joblib.dump") as mock_dump:
        metrics = train_and_evaluate(
            "logreg", model, X_train, X_test, y_train, y_test
        )

    # Verify dictionary keys
    expected_keys = {
        "model", "Accuracy", "Precision", "Recall", "F1 Score",
        "TN(1,1)", "FP(1,2)", "FN(2,1)", "TP(2,2)"
    }
    assert set(metrics.keys()) == expected_keys
    assert metrics["model"] == "logreg"
    assert 0.0 <= metrics["Accuracy"] <= 1.0

    # Ensure joblib.dump was called
    mock_dump.assert_called_once()


def test_run_ml_models_returns_dataframe(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    results = run_ml_models(X_train, X_test, y_train, y_test)

    # Check type and structure
    assert isinstance(results, pd.DataFrame)
    assert not results.empty
    assert "model" in results.columns
    assert "Accuracy" in results.columns
    assert results["Accuracy"].between(0.0, 1.0).all()


def test_run_ml_models_contains_all_models(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    results = run_ml_models(X_train, X_test, y_train, y_test)
    expected_models = {
        "svc1", "svc2", "gnb", "dtc1", "dtc2",
        "rfc1", "rfc2", "rfc3", "knn1", "knn2", "knn3"
    }

    assert set(results["model"]) == expected_models
