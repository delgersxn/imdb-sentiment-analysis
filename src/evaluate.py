from sklearn.metrics import classification_report
import joblib


def evaluate_model(X_texts, y_true):
    vectorizer = joblib.load("model/vectorizer.joblib")
    clf = joblib.load("model/classifier.joblib")

    X = vectorizer.transform(X_texts)
    y_pred = clf.predict(X)

    print(classification_report(y_true, y_pred))
