from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import os


def train_model(X_texts, y_labels):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_texts)

    clf = MultinomialNB()
    clf.fit(X, y_labels)

    os.makedirs("model", exist_ok=True)
    joblib.dump(vectorizer, "model/vectorizer.joblib")
    joblib.dump(clf, "model/classifier.joblib")

    return clf, vectorizer
