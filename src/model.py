import numpy as np
import pandas as pd

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
import joblib


def train_and_evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
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
    df = load_data("data/heart.csv")
    X_train, X_test, y_train, y_test, encoder, cols_to_encode, scaler, pca = preprocess_data(df)
    results = run_ml_models(X_train, X_test, y_train, y_test)
    results.to_csv("results/model_performance.csv", index=False)
    print(results.head(10))
    


