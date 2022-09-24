from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy
import sklearn

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('web.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST': 
        
        age = int(request.form['age'])
        trtbps = float(request.form['trtbps'])
        chol = float(request.form['chol'])
        
        thalach = float(request.form['thalach'])        
        oldpeak = float(request.form['oldpeak'])
        
        sex = request.form['sex']
        if (sex == "M"):
            sex = 1
        else:
            sex = 0
            
        fbs = request.form['fbs'] 
        if (fbs=="1"):
            fbs = 1
        else:
            fbs = 0
        
        exang = request.form['exang']
        if (exang=="1"):
            exang = 1
        else:
            exang = 0
        
        slp = request.form['slp']
        if (slp=="1"):
            slp = 1
        elif (slp=="2"):
            slp = 2
        else:
            slp = 0
        
        caa = request.form['caa']
        if (caa=="1"):
            caa = 1
        elif (caa=="2"):
            caa = 2
        elif (caa=="3"):
            caa = 3
        elif (caa=="4"):
            caa = 4
        else:
            caa = 0
        
        thall = request.form['thall']
        if (thall=="1"):
            thall = 1
        elif (thall=="2"):
            thall = 2
        elif (thall=="3"):
            thall = 3
        else:
            thall = 0
        
        cp = request.form['cp']
        if(cp == "0"):
            cp_0 = 1
            cp_1 = 0
            cp_2 = 0
            cp_3 = 0
        elif (cp == "1"):
            cp_0 = 0
            cp_1 = 1
            cp_2 = 0
            cp_3 = 0
        elif (cp == "2"):
            cp_0 = 0
            cp_1 = 0
            cp_2 = 1
            cp_3 = 0
        else:
            cp_0 = 0
            cp_1 = 0
            cp_2 = 0
            cp_3 = 1
        
        restecg = request.form['restecg']
        if(restecg == "0"):
            restecg_1 = 0
            restecg_2 = 0
        elif (restecg == "1"):
            restecg_1 = 1
            restecg_2 = 0
        else:
            restecg_1 = 0
            restecg_2 = 1
            
        prediction=model.predict([[age, sex, trtbps, chol, fbs, thalach, exang,
                                   oldpeak, slp, caa, thall, cp_0, cp_1, cp_2, 
                                   cp_3, restecg_1, restecg_2]])
        
        
        if prediction[0] == 1:
            return render_template('result1.html')
        else:
            return render_template('result0.html')
        
        
    else:
        return render_template('web.html', prediction_text="Hello")
    
    

if __name__=="__main__":
    app.run(debug=True)

