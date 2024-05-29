from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import pickle

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("homepage.html")

@app.route('/hospital')
def hospital():
    return render_template("hospital.html")

@app.route('/bmi')
def bmi():
    return render_template("bmi.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route("/result", methods=['POST', 'GET'])
def result():
    
    if request.method == 'POST':
        # Convert string gender values to integer
        gender = {'Female': 0, 'Male': 1}[request.form['gender']]

        # Convert string marital status values to integer
        ever_married = {'married': 1, 'not married': 0}[request.form['maritalstatus']]

        # Convert string work type values to integer
        work_type = {'privatejob': 2, 'govtemp': 0, 'selfemp': 3 ,'neverworked':1}[request.form['Worktype']]

        # Convert string residence type values to integer
        Residence_type = {'urban': 1, 'rural': 0}[request.form['Residence']]

        # Convert string smoking status values to integer
        smoking_status = {'formerly-smoked': 1, 'yes-smokes': 3, 'never-smoked': 2}[request.form['Smoke']]

        # Convert string hypertension values to integer
        hypertension = {'hypten': 1, 'nohypten': 0}[request.form['Hypertension']]

        # Convert string heart disease values to integer
        heart_disease = {'heartdis': 1, 'noheartdis': 0}[request.form['Heartdisease']]
        

        age = int(request.form['age'])
        avg_glucose_level = float(request.form['gluclevel'])
        bmi = float(request.form['bmi'])
        

        x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

        scalar_path = os.path.join('D:\AIHT FINAL PROJECT\Final Webpages\Early-Detection-of-Brain-Stroke-main', 'models/scalar.pkl')
        #scalar_path = os.path.join('/', 'models/scalar.pkl')

        scalar = None
        with open(scalar_path, 'rb') as scalar_file:
            scalar = pickle.load(scalar_file)

        x = scalar.transform(x)

        model_path = os.path.join('D:\AIHT FINAL PROJECT\Final Webpages\Early-Detection-of-Brain-Stroke-main', 'models/model.sav')
        model = joblib.load(model_path)

        y_pred = model.predict(x)

        if y_pred == 0:
            return render_template('nostroke.html')
        else:
            return render_template('stroke.html')   
    else:
        return render_template('detection.html')    
if __name__ == "__main__":
   app.run(debug=True)

