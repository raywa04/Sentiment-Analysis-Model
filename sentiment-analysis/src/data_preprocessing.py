import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    return X, y, vectorizer

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
