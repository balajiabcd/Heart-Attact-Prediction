def add_features(df):
    # Example: create interaction features
    df['age_chol_ratio'] = df['age'] / df['chol']
    return df
