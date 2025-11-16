from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

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
    app.run(debug=True)
