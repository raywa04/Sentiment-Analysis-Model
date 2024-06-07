from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from data_preprocessing import load_data, preprocess_data, split_data

def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy}')
    return accuracy

if __name__ == "__main__":
    df = load_data('data/tweets.csv')
    X, y, vectorizer = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    
    joblib.dump(model, 'models/sentiment_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
