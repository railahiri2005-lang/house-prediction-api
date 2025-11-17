import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # 🔥 Allow requests from your HTML file

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "🏡 House Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['area'], data['bedrooms'], data['bathrooms']]])
    prediction = model.predict(features)
    return jsonify({"Predicted Price": float(prediction[0])})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
