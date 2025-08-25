import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.feature_engineering import prepare_features, perform_pca
from sklearn.preprocessing import OneHotEncoder





def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data((df))

    X_train, X_test, encoder, cols_to_encode = perform_encoding(X_train, X_test)
    X_train, X_test, scaler = perform_standardization(X_train, X_test)
    X_train, X_test, pca = perform_pca(X_train, X_test)

    return X_train, X_test, y_train, y_test, encoder, cols_to_encode, scaler, pca





def split_data(df: pd.DataFrame):
    Y = df.output
    X = df.drop("output", axis =1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test





def perform_encoding(X_train, X_test):
    encoder, cols_to_encode = encode(X_train)
    X_train = transform_with_saved_encoder(X_train, encoder, cols_to_encode)
    X_test = transform_with_saved_encoder(X_test, encoder, cols_to_encode)
    return X_train, X_test, encoder, cols_to_encode

def encode(df):
    cols_to_encode = [
        col for col in df.columns
        if df[col].nunique() > 2 and df[col].nunique() < 8]
    encoder = OneHotEncoder(sparse=False, drop=None, handle_unknown="ignore")
    encoder = encoder.fit(df[cols_to_encode])
    return encoder, cols_to_encode

def transform_with_saved_encoder(df, encoder, cols_to_encode):
    encoded_array = encoder.transform(df[cols_to_encode])
    encoded_cols = encoder.get_feature_names_out(cols_to_encode)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df_other = df.drop(columns=cols_to_encode)
    final_df = pd.concat([df_other, encoded_df], axis=1)
    return final_df





def perform_standardization(X_train, X_test):
    scaler = standardize_data(X_train)
    X_train = transform_with_saved_scaler(X_train, scaler)
    X_test = transform_with_saved_scaler(X_test, scaler)
    return X_train, X_test, scaler

def standardize_data(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit(X_train)
    return scaler

def transform_with_saved_scaler(X, scaler):
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled_df

