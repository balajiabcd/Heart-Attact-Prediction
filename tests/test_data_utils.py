# tests/test_data_utils.py

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src import data_utils


@pytest.fixture
def sample_df():
    df = pd.DataFrame({
        "age": [25, 30, 40, 35, 50],
        "gender": ["M", "F", "M", "F", "M"],
        "cholesterol": [200, 180, 190, 210, 220],
        "output": [1, 0, 1, 0, 1]
    })
    df["gender"] = df["gender"].map({"M":0, "F":1})  # convert to numeric
    return df


def test_load_data(tmp_path):
    """Test loading data from a CSV file."""
    csv_file = tmp_path / "test.csv"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(csv_file, index=False)

    loaded = data_utils.load_data(csv_file)
    pd.testing.assert_frame_equal(df, loaded)


def test_split_data(sample_df):
    """Test splitting data into train/test sets."""
    X_train, X_test, y_train, y_test = data_utils.split_data(sample_df)

    assert "output" not in X_train.columns
    assert "output" not in X_test.columns
    assert len(X_train) + len(X_test) == len(sample_df)
    assert len(y_train) + len(y_test) == len(sample_df)


def test_encode_and_transform(sample_df):
    """Test encoding categorical features."""
    encoder, cols_to_encode = data_utils.encode(sample_df.drop(columns="output"))
    assert isinstance(encoder, OneHotEncoder)
    assert "gender"  not in cols_to_encode

    transformed = data_utils.transform_with_saved_encoder(
        sample_df.drop(columns="output"), encoder, cols_to_encode
    )
    for enc_col in encoder.get_feature_names_out(): 
        assert enc_col in transformed.columns


def test_perform_encoding(sample_df):
    """Test encoding wrapper function."""
    X = sample_df.drop(columns="output")
    X_train, X_test, encoder, cols = data_utils.perform_encoding(X, X)
    assert isinstance(encoder, OneHotEncoder)
    assert all(col in X_train.columns for col in encoder.get_feature_names_out(cols))


def test_standardization(sample_df):
    """Test standardization and transformation."""
    X = sample_df.drop(columns="output")
    scaler = data_utils.standardize_data(X)
    assert isinstance(scaler, StandardScaler)

    X_scaled = data_utils.transform_with_saved_scaler(X, scaler)
    np.testing.assert_almost_equal(X_scaled.mean().values.round(0), np.zeros(len(X_scaled.columns)))


def test_perform_standardization(sample_df):
    """Test standardization wrapper function."""
    X = sample_df.drop(columns="output")
    X_train, X_test, scaler = data_utils.perform_standardization(X, X)
    assert isinstance(scaler, StandardScaler)
    assert np.allclose(X_train.std(ddof=0).values, 1, atol=1e-6)


def test_preprocess_data(monkeypatch, sample_df):
    """Test preprocess_data with mocked feature engineering & plots."""

    # Mock dependencies to avoid external calls
    monkeypatch.setattr("src.data_utils.prepare_features", lambda df: df)
    monkeypatch.setattr("src.data_utils.perform_pca", lambda X_train, X_test: (X_train, X_test, None))
    monkeypatch.setattr("src.data_utils.plots_data_analysis", lambda df: None)
    monkeypatch.setattr("src.data_utils.plots_all_features", lambda df: None)

    X_train, X_test, y_train, y_test, encoder, cols, scaler, pca = data_utils.preprocess_data(sample_df)

    # Assertions
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert len(X_train) > 0 and len(X_test) > 0
    assert encoder is not None
    assert scaler is not None
