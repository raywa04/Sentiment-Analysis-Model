# Sentiment Analysis on Social Media

## Project Description

This project aims to build a machine learning model to analyze the sentiment of social media posts (e.g., Twitter, Facebook). The model will classify posts as positive, negative, or neutral.

## Features

1. Data collection from social media APIs.
2. Preprocessing text data (tokenization, stop-word removal).
3. Training and evaluating sentiment analysis models (e.g., using Naive Bayes).

## Technology Stack

- Python
- Flask
- Scikit-learn
- Pandas
- Joblib

## Setup Instructions

1. Install Python and pip.
2. Navigate to the `sentiment-analysis` folder.
3. Install dependencies: `pip install -r requirements.txt`.
4. Run the model training script: `python src/model_training.py`.
5. Start the Flask app: `python src/app.py`.

## Usage

Send a POST request to `/predict` with JSON data containing the text to be analyzed.

Example:
```bash
curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"text": "I love this product!"}'
```
