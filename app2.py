from flask import Flask, render_template, request
import pickle
import pandas as pd
from src.data_utils import preprocess_data

app = Flask(__name__)
model = pickle.load(open('models/rfc3.pkl', 'rb'))

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

    df = pd.DataFrame(data, index=[0])
    df = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data((df))
    df_fe = X_train.copy()
    df_fe["output"] = y_train.values
    plots_all_features(df_fe)

    X_train, X_test, encoder, cols_to_encode = perform_encoding(X_train, X_test)
    X_train, X_test, scaler = perform_standardization(X_train, X_test)
    X_train, X_test, pca = perform_pca(X_train, X_test)
    
    prediction=model.predict(df).tolist()

    if prediction[0] == 0:
        return render_template("result_low.html")
    else:
        return render_template("result_high.html")

    # return prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


