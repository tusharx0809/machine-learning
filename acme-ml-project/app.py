from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load the model, scaler, and encoder
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form inputs
    age = float(request.form['age'])
    bmi = float(request.form['bmi'])
    children = int(request.form['children'])
    smoker = request.form['smoker']
    sex = request.form['sex']
    region = request.form['region']

    # Process categorical inputs
    smoker_code = 1 if smoker == 'yes' else 0
    sex_code = 1 if sex == 'male' else 0

    # One-hot encode region
    region_encoded = encoder.transform([[region]]).toarray()

    # Scale numeric features
    scaled_numeric = scaler.transform([[age, bmi, children]])

    # Combine all inputs
    inputs = np.concatenate((scaled_numeric, [[smoker_code, sex_code]], region_encoded), axis=1)

    # Make prediction
    prediction = model.predict(inputs)[0]

    return render_template('index.html', prediction_text='Predicted Charges: INR {:.2f}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)
