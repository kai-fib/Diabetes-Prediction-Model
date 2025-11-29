
from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import sys
# --- scikit-learn pickle compatibility patch ---
# Older models used sklearn.ensemble.forest, which no longer exists.
# We map it to the new location so pickle can find it.
import sklearn.ensemble._forest as _forest
sys.modules['sklearn.ensemble.forest'] = _forest
# ------------------------------------------------

# filename = 'diabetes-prediction-rfc-model.pkl'
# classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)
# Build safe absolute path to the model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'diabetes-prediction-rfc-model.pkl')

with open(model_path, 'rb') as f:
    classifier = pickle.load(f)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)