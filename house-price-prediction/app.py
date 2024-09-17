from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np


model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form['square_footage']),
                float(request.form['num_bedrooms']),
                float(request.form['num_bathrooms']),
                float(request.form['year_built']),
                float(request.form['lot_size']),
                float(request.form['garage_size']),
                float(request.form['neighborhood_quality'])]
    
    prediction = model.predict([np.array(features)])
    return render_template('index.html', prediction=round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)
