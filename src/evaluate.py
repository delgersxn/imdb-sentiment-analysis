from sklearn.metrics import classification_report
import joblib                                    # for loading saved models

def evaluate_model(X_texts, y_true):           
    vectorizer = joblib.load("model/vectorizer.joblib") # Loads the saved vectorizer and remembers vocabulary from training .fit()
    clf = joblib.load("model/classifier.joblib")        # Loads the saved classifier

    X = vectorizer.transform(X_texts)            # Transforms the test texts using the vectorizer
    y_pred = clf.predict(X)                      # Predicts sentiment labels for the test data

    print(classification_report(y_true, y_pred))
    # precision: of all the reviews the model said were positive, how many really were positive?
    # recall: of all the truly positive reviews, how many did the model correctly find?
    # f1-score: harmonic mean of precision and recall, useful for imbalanced datasets
    # support: number of true instances for each class in the test set
