from flask import Flask, request, app, jsonify,url_for, render_template, redirect, flash, session
from markupsafe import escape
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

app = Flask(__name__)

# Load the model
model = joblib.load(open('final_model.pkl', 'rb'))

@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)

    # Convert incoming data to array
    new_data = np.array(list(data.values())).reshape(1, -1)

    # If your model did NOT use scaling during training, do not scale here
    # new_data = scaler.transform(new_data)

    # Get probabilities (probability for class 1 is at index [:, 1])
    y_proba = model.predict_proba(new_data)[:, 1]

    # Apply custom threshold
    y_pred = (y_proba >= 0.3).astype(int)

    print(f"Probability: {y_proba[0]:.4f}")
    print(f"Prediction: {y_pred[0]}")

    return jsonify(int(y_pred[0]))

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    new_data = np.array(data).reshape(1,-1)
    print(new_data)
    # Get probabilities (probability for class 1 is at index [:, 1])
    y_proba = model.predict_proba(new_data)[:, 1]

    # Apply custom threshold
    y_pred = (y_proba >= 0.3).astype(int)
    if y_pred == 0:
        y_pred = 'You dont have Heart Failure'
    else:
        y_pred = 'You have Heart Failure'
    return render_template('home.html', prediction_text = 'The result is {}'.format(y_pred))


if __name__ == "__main__":
    app.run(debug=True)