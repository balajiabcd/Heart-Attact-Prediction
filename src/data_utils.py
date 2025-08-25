

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna()
    df, encoder, cols_to_encode = encode(df)
    X_train, X_test, y_train, y_test = split_data((df))
    X_train, X_test, scaler = standardize_data(X_train, X_test)
    return X_train, X_test, y_train, y_test, encoder, cols_to_encode, scaler

def encode(df):
    cols_to_encode = [
        col for col in df.columns
        if df[col].nunique() > 2 and df[col].nunique() < 8]
    encoder = OneHotEncoder(sparse=False, drop=None)
    encoded_array = encoder.fit_transform(df[cols_to_encode])
    encoded_cols = encoder.get_feature_names_out(cols_to_encode)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df_other = df.drop(columns=cols_to_encode)
    final_df = pd.concat([df_other, encoded_df], axis=1)
    return final_df, encoder, cols_to_encode

def transform_with_saved_encoder(df, encoder, cols_to_encode):
    encoded_array = encoder.transform(df[cols_to_encode])
    encoded_cols = encoder.get_feature_names_out(cols_to_encode)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
    df_other = df.drop(columns=cols_to_encode)
    final_df = pd.concat([df_other, encoded_df], axis=1)
    return final_df

def split_data(df: pd.DataFrame):
    Y = df.output
    X = df.drop("output", axis =1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def standardize_data(X_train, X_test"):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=df.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns, index=df.index)
    return X_train_scaled_df, X_test_scaled_df, scaler

def transform_with_saved_scaler(X, scaler):
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    return X_scaled_df

