import joblib
import pandas as pd
from flask import Flask, render_template, request
from src.feature_engineering import prepare_features
from src.data_utils import transform_with_saved_encoder

app = Flask(__name__)
model = joblib.load("models/rfc3.pkl")

@app.route('/')
def Home():
    return render_template("normal.html")

@app.route('/predict',methods=['POST'])



def hell():
    data = {
    'age': int(request.form['age']),
    'sex': int(request.form['sex']),
    'cp': request.form['cp'],
    'trtbps': float(request.form['trtbps']),
    'chol': float(request.form['chol']),
    'fbs': int(request.form['fbs']),
    'restecg': request.form['restecg'],
    'thalachh': float(request.form['thalach']),
    'exang': int(request.form['exang']),
    'oldpeak': float(request.form['oldpeak']),
    'slp': int(request.form['slp']),
    'caa': int(request.form['caa']),
    'thall': int(request.form['thall'])
    }

    testdf = pd.DataFrame(data, index=[0])
    encoder1 = joblib.load("models/encoder.pkl")
    cols_to_encode1 = joblib.load("models/cols_to_encode.pkl")
    scaler1 = joblib.load("models/scaler.pkl")
    pca1 = joblib.load("models/pca.pkl")
    model1 = joblib.load("models/best_model.pkl")

    ALIAS_MAP = {
    "trestbps": "trtbps",
    "thalach": "thalachh",
    "exang": "exng",}

    def normalize_input_columns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=ALIAS_MAP)
        return df

    testdf = normalize_input_columns(testdf)
    testdf = prepare_features(testdf)
    testdf_encoded = transform_with_saved_encoder(testdf, encoder1, cols_to_encode1)
    df_standardized = scaler1.transform(testdf_encoded)
    testdf_encoded_pca = pca1.transform(df_standardized)
    result = model1.predict(testdf_encoded_pca)

    if result[0] == 0:
        return render_template("result_low.html")
    else:
        return render_template("result_high.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)