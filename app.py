from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('ml_model.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template("normal.html")
    #return "Balaji"

@app.route('/predict',methods=['POST'])

#def create_dataframe(keys,vls):
#    return pd.DataFrame(vls, index=keys).T

def hell():
    age = int(request.form['age'])
    trtbps = float(request.form['trtbps'])
    chol = float(request.form['chol'])
    thalach = float(request.form['thalach'])        
    oldpeak = float(request.form['oldpeak'])
    sex = int(request.form['sex'])
    fbs = int(request.form['fbs'])
    exang = int(request.form['exang'])
    slp = int(request.form['slp'])
    caa = int(request.form['caa'])
    thall = int(request.form['thall'])

    cp_1 = 0
    cp_2 = 0
    cp_3 = 0
    cp = request.form['cp']
    if (cp == "1"): cp_1 = 1
    if (cp == "2"): cp_2 = 1
    if (cp == "3"): cp_3 = 1

    restecg_1 = 0
    restecg_2 = 0
    restecg = request.form['restecg']
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
    app.run(debug=True)


