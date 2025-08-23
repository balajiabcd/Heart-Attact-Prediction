import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values
    df = df.dropna()

    '''

Handling categorical variables: Encoding (one-hot, label encoding, target encoding).

Scaling/Normalization: StandardScaler, MinMaxScaler for algorithms sensitive to scale (SVM, KNN, neural networks).

Data splitting: Train-test split or cross-validation setup.

Feature type conversion: Convert dates, strings, or other types into numerical or categorical formats.

Removing duplicates: Ensure data is clean and non-redundant.'''


    
    Y = df.output
    X = df.drop("output", axis =1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def split_data(df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(target, axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
