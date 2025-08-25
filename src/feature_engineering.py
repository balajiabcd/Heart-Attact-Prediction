import pandas as pd
import numpy as np

def feature_engineering(df):
    df = add_risk_flags_and_bins(df)
    df = add_interactions_and_polynomials(df)
    df = risk_scores(df)
    return df

def add_risk_flags_and_bins(df):
    # imp for medical analysis and domain knowledge
    df["is_hypertensive"] = (df["trtbps"] >= 140).astype(int)
    df["is_hyperchol"] = (df["chol"] >= 240).astype(int)
    df["is_oldpeak_high"] = (df["oldpeak"] >= 2.0).astype(int)
    
    # Binning some variables into categories
    # Age groups
    df["age_group"] = pd.cut(df["age"],
                             bins=[0, 40, 60, np.inf],
                             labels=["young", "middle_aged", "elderly"])
    # Cholesterol groups
    df["chol_group"] = pd.cut(df["chol"],
                              bins=[0, 200, 240, np.inf],
                              labels=["normal", "borderline", "high"])
    # Blood pressure groups
    df["bp_group"] = pd.cut(df["trtbps"],
                            bins=[0, 120, 139, np.inf],
                            labels=["normal", "elevated", "hypertension"])
    # Max heart rate groups
    df["thalachh_group"] = pd.cut(df["thalachh"],
                                  bins=[0, 100, 140, np.inf],
                                  labels=["low_fitness", "average", "good"])
    return df


def add_interactions_and_polynomials(df):
    # some new features based on domain knowledge
    df["age_chol_ratio"] = df["age"] / (df["chol"] + 1)       # +1 is to avoid div by zero
    df["age_bp_ratio"] = df["age"] / (df["trtbps"] + 1)
    df["chol_hdl_ratio"] = df["chol"] / (df["thalachh"] + 1)  # cholesterol relative to max HR
    df["bp_chol_product"] = df["trtbps"] * df["chol"]         # interaction effect
    df["bp_oldpeak_interaction"] = df["trtbps"] * df["oldpeak"]
    # some non-linear transformations
    df["age_squared"] = df["age"] ** 2
    df["chol_squared"] = df["chol"] ** 2
    df["oldpeak_squared"] = df["oldpeak"] ** 2
    df["bp_squared"] = df["trtbps"] ** 2
    # Log transforms (only on positive values)
    df["log_chol"] = np.log1p(df["chol"])
    df["log_trtbps"] = np.log1p(df["trtbps"])
    df["log_oldpeak"] = np.log1p(df["oldpeak"].clip(lower=0))  # clip avoids log(-ve)
    # Square roots
    df["sqrt_age"] = np.sqrt(df["age"])
    df["sqrt_chol"] = np.sqrt(df["chol"].clip(lower=0))
    df["sqrt_trtbps"] = np.sqrt(df["trtbps"].clip(lower=0))
    return df


def risk_scores(df):
    # Count how many individual risks are present
    df["risk_count"] = (
        (df["trtbps"] >= 140).astype(int) +       # hypertension
        (df["chol"] >= 240).astype(int) +         # high cholesterol
        (df["fbs"] == 1).astype(int) +            # diabetes
        (df["exng"] == 1).astype(int) +           # exercise induced angina
        (df["oldpeak"] >= 2).astype(int)          # ST depression
    )
    # --- Weighted composite score (example weights)
    df["weighted_risk_score"] = (
        0.3 * (df["trtbps"] / 180) +        # normalized BP
        0.3 * (df["chol"] / 300) +          # normalized cholesterol
        0.2 * (df["oldpeak"] / 5) +         # ST depression
        0.1 * df["fbs"] +                   # diabetes
        0.1 * df["exng"]                    # exercise angina
    )
    df["cardiac_stress_index"] = (220 - df["age"]) - df["thalachh"]
    # --- Cholesterol to age ratio ---
    df["chol_age_ratio"] = df["chol"] / (df["age"] + 1)
    # --- Combined BP & cholesterol risk ---
    df["bp_chol_risk"] = (df["trtbps"] / 120) + (df["chol"] / 200)
    return df


