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
    X_train, X_test, y_train, y_test, encoder, cols_to_encode, scaler, pca = preprocess_data(df)
    



    cp_1 = 0
    cp_2 = 0
    cp_3 = 0
    if (cp == "1"): cp_1 = 1
    if (cp == "2"): cp_2 = 1
    if (cp == "3"): cp_3 = 1

    restecg_1 = 0
    restecg_2 = 0
    if (restecg == "1"):    restecg_1 = 1
    if (restecg == "2"):    restecg_2 = 1

    keys =  ['age', 'sex', 'trtbps', 'chol',  'fbs',  'thalachh', 'exng',      'oldpeak',
             'slp', 'caa', 'thall',   'cp_1', 'cp_2', 'cp_3',     'restecg_1', 'restecg_2']

    values =[ age,   sex,   trtbps,    chol,   fbs,    thalach,   exang,         oldpeak, 
              slp,   caa,    thall,    cp_1,   cp_2,     cp_3,   restecg_1,      restecg_2]

    #df = create_dataframe(keys, values)
    #prediction=model.predict(df)


    #return f"age = {age}, rtbps = {trtbps}, chol = {chol}, thalach = {thalach}, oldpeak = {oldpeak}, sex = {sex}, fbs = {fbs}, exang = {exang}, slp = {slp}, caa={caa}, thall={thall}, cp_1 = {cp_1}, restecg_1 = {restecg_1}, " #prediction = {prediction}
    dictionary = {'age': age, 'sex': sex, 'trtbps':trtbps, 'chol': chol,  'fbs': fbs,  'thalachh': thalach, 'exng': exang, 'oldpeak': oldpeak,  'slp': slp, 'caa':caa, 'thall':thall,   'cp_1':cp_1, 'cp_2': cp_2, 'cp_3': cp_3, 'restecg_1': restecg_1, 'restecg_2': restecg_2}
    df = pd.DataFrame(dictionary, index=[0]) 
    prediction=model.predict(df).tolist()

    if prediction[0] == 0:
        return render_template("result_low.html")
    else:
        return render_template("result_high.html")

    # return prediction


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)


