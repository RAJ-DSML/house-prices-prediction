import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__)

# Load the model
reg_model = pickle.load(open('reg_model.pkl', 'rb'))

# Load the scaling data model
scaling_model = pickle.load(open('scaling_data.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))

    new_data = scaling_model.transform(np.array(list(data.values())).reshape(1, -1))
    output = reg_model.predict(new_data)

    print(output[0])

    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaling_model.transform(np.array(data).reshape(1, -1))

    print(final_input)

    output = reg_model.predict(final_input)[0]

    return render_template("home.html", prediction_text = "The Predicted House Price is: {}".format(output))


if __name__ == "__main__":
    app.run(debug=True)