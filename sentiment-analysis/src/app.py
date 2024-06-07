from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('models/sentiment_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['text']
    transformed_data = vectorizer.transform([data])
    prediction = model.predict(transformed_data)
    return jsonify({'sentiment': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
