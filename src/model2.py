import numpy as np
import pandas as pd
import os
import joblib

from src.data_utils import load_data, preprocess_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score,  f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)





def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    if not os.path.exists("models"):
        os.makedirs("models")

    joblib.dump(model, "models/"+name+".pkl")
    y_pred = model.predict(X_test)
    #y_proba = model.predict_proba(X_test)[:, 1] 
    
    print("model name: ",name)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    print("F1 Score: ", f1_score(y_test, y_pred))
    #print("ROC-AUC: ", roc_auc_score(y_test, y_proba))
    #print("PR-AUC: ", average_precision_score(y_test, y_proba))
    print("\nConfusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("___________________________________________________________________________")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
    "model": name,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    'TN(1,1)': tn,
    'FP(1,2)': fp,  
    'FN(2,1)': fn,
    'TP(2,2)': tp
    #"ROC-AUC": roc_auc_score(y_test, y_proba),
    #"PR-AUC": average_precision_score(y_test, y_proba)
    }
    return metrics

def run_ml_models(X_train, X_test, y_train, y_test):
    models = [
    ("svc1", SVC(random_state=2)),
    ("svc2", SVC(kernel="rbf", random_state=2)),
    ("gnb", GaussianNB()),
    ("dtc1", DecisionTreeClassifier(random_state=2)),
    ("dtc2", DecisionTreeClassifier(random_state=2, criterion="entropy")),
    ("rfc1", RandomForestClassifier(random_state=2, n_estimators=10, criterion="gini")),
    ("rfc2", RandomForestClassifier(random_state=2, n_estimators=10, criterion="entropy")),
    ("rfc3", RandomForestClassifier(random_state=2, n_estimators=400, criterion="entropy")),
    ("knn1", KNeighborsClassifier(p=2, n_neighbors=5)),
    ("knn2", KNeighborsClassifier(p=2, n_neighbors=15)),
    ("knn3", KNeighborsClassifier(p=2, n_neighbors=25))
    ]
    
    results = []
    for name, model in models:
        metrics = train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
        results.append(metrics)
    return pd.DataFrame(results)


if __name__ == "__main__":
    from pathlib import Path
    import os, joblib, shutil

    # Resolve repo root:  <repo>/  (since this file is <repo>/src/model.py)
    ROOT = Path(__file__).resolve().parent.parent
    DATA_PATH = ROOT / "data" / "heart.csv"
    ARTIFACTS_DIR = ROOT / "artifacts"
    MODELS_DIR = ROOT / "models"
    RESULTS_DIR = ROOT / "results"

    # Ensure folders exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load & preprocess
    df = load_data(str(DATA_PATH))
    X_train, X_test, y_train, y_test, encoder, cols_to_encode, scaler, pca = preprocess_data(df)

    # 2) Save preprocessing artifacts
    joblib.dump(encoder, ARTIFACTS_DIR / "encoder.pkl")
    joblib.dump(cols_to_encode, ARTIFACTS_DIR / "cols_to_encode.pkl")
    joblib.dump(scaler, ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(pca, ARTIFACTS_DIR / "pca.pkl")

    # 3) Train all models and save metrics
    results = run_ml_models(X_train, X_test, y_train, y_test)
    results.to_csv(RESULTS_DIR / "model_performance.csv", index=False)

    # 4) Copy the best model to a canonical filename (optional, nice for Flask)
    best = results.sort_values("F1 Score", ascending=False).iloc[0]["model"]
    src = MODELS_DIR / f"{best}.pkl"
    dst = MODELS_DIR / "best_model.pkl"
    shutil.copyfile(src, dst)

    # 5) Debug prints so you can see where files landed
    print("\nArtifacts saved to:", ARTIFACTS_DIR)
    for p in ARTIFACTS_DIR.glob("*.pkl"): print("  -", p.name)
    print("Models saved to:", MODELS_DIR)
    for p in MODELS_DIR.glob("*.pkl"): print("  -", p.name)
    print("Results saved to:", RESULTS_DIR / "model_performance.csv")
    print(results.head(10))


    


