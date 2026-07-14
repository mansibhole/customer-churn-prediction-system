from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler

model = pickle.load(open('models/churn_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    tenure = float(request.form['tenure'])
    monthlycharges = float(request.form['monthlycharges'])
    totalcharges = float(request.form['totalcharges'])

    input_data = np.array([
        tenure,
        monthlycharges,
        totalcharges
    ]).reshape(1, -1)

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        result = 'Customer Will Churn'
    else:
        result = 'Customer Will Stay'

    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)