import pickle
from flask import Flask, request, render_template, json, jsonify, app, url_for
import pandas as pd
import numpy as np

##loading model and scaler
model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    input = scaler.transform(np.array(list(data.values)).reshape(1, -1))
    output = model.predict(input)
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    input = scaler.transform(np.array(list(data)).reshape(1, -1))
    output = model.predict(input)[0]
    return render_template('home.html', prediction_text = f'The predicted house price is {output}')


if __name__ == "__main__":
    app.run(debug=True)