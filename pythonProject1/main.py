from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

#loading the model using joblib
try:
    model = joblib.load('best_model_.pkl')
    print('Model loaded successfully using joblib!')
except Exception as e:
    print('Error loading model using joblib:', e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/index')
def form():
    return render_template('/Users/jkottu/PycharmProjects/pythonProject1/index.html')


def predict_cad(male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP ,diaBP, BMI, heartRate, glucose):

    array_features = [np.array([male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose])]

    # Example feature names
    feature_names = ['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds',
                     'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
                     'diaBP', 'BMI', 'heartRate', 'glucose']

    # Create DataFrame from array with column names
    input_data = pd.DataFrame(data=array_features, columns=feature_names)
    prediction = model.predict(input_data)
    output = prediction

    # Check the output values and retrieve the result with html tag based on the value
    return 'The patient is likely to have heart disease!' if output == 1 else 'The patient is not likely to have heart disease!'


@app.route('/', methods=['POST'])
def predict():

    male = int(request.form['male'])
    age = int(request.form['age'])
    education = int(request.form['education'])
    currentSmoker = int(request.form['currentSmoker'])
    cigsPerDay = int(request.form['cigsPerDay'])
    BPMeds = int(request.form['BPMeds'])
    prevalentStroke = int(request.form['prevalentStroke'])
    prevalentHyp = int(request.form['prevalentHyp'])
    diabetes = float(request.form['diabetes'])
    totChol = int(request.form['totChol'])
    sysBP = float(request.form['sysBP'])
    diaBP = float(request.form['diaBP'])
    BMI = float(request.form['BMI'])
    heartRate = int(request.form['heartRate'])
    glucose = int(request.form['glucose'])

    #Make the prediction
    prediction = predict_cad(male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose)

    return render_template('results.html', prediction=prediction)


if __name__ == '__main__':
    app.run()

