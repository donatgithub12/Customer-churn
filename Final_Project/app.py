from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def welcome():
    return render_template('welcome2.html')

@app.route('/input')
def input():
    return render_template('input2.html')

@app.route('/predict', methods=['POST'])
def predict():
    Age = float(request.form['Age'])
    Total_Purchase = float(request.form['Total_Purchase'])
    Account_Manager = int(request.form['Account_Manager'])
    Years = float(request.form['Years'])
    Num_Sites = float(request.form['Num_Sites'])
    
    prediction = model.predict([[Age, Total_Purchase, Account_Manager, Years, Num_Sites]])
    return render_template('result2.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
