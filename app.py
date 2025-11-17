from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)   # allow HTML/JS access

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return "House Price Prediction API is running!"

# Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    area = float(data['area'])
    bedrooms = int(data['bedrooms'])
    bathrooms = int(data['bathrooms'])

    # Prepare input
    features = np.array([[area, bedrooms, bathrooms]])

    # Predict
    prediction = model.predict(features)[0]

    # Avoid negative outputs
    if prediction < 0:
        prediction = 0

    return jsonify({"Predicted Price": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)
