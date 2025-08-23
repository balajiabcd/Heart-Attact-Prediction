import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values
    df = df.dropna()

    # Encoding categorical variables
    df = encode(df)

    # Standardization/Normalization of numerical features.
    df = standardize_data(df)

    # Data splitting: Train-test split or cross-validation setup.
    X_train, X_test, y_train, y_test = split_data((df))

    return X_train, X_test, y_train, y_test

def encode(df):

    cols_to_encode = [ col for col in df.columns
                    if df[col].nunique() > 2 and df[col].nunique() < 8]
                    
    df_encoded = pd.get_dummies(df, columns=cols_to_encode, prefix=cols_to_encode, prefix_sep="_")*1
    return df_encoded




def standardize_data(df, target_col="output"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=df.index)
    result_df = pd.concat([X_scaled_df, y], axis=1)
    return result_df

def split_data(df: pd.DataFrame):
    Y = df.output
    X = df.drop("output", axis =1)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
