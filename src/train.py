from sklearn.feature_extraction.text import CountVectorizer   # Imports CountVectorizer for converting text to feature vectors
from sklearn.naive_bayes import MultinomialNB                # Imports the Multinomial Naive Bayes classifier
import joblib                                                # Imports joblib for saving Python objects to disk
import os                                                    # Imports os for file and directory operations

def train_model(X_texts, y_labels):                          # Defines a function to train the model (review, sentiment)
    vectorizer = CountVectorizer()                           # Creates a CountVectorizer instance (bag-of-words) 
    # vectorizer: count number of times each word appears, and creates a table of word counts <- feature vector
    X = vectorizer.fit_transform(X_texts)                    # Learns the vocabulary and transforms texts to feature vectors

    clf = MultinomialNB()                                    # Creates a Multinomial Naive Bayes classifier
    # classifier: learns the probabilities of each word given a class to predict sentiment (positive/negative)
    clf.fit(X, y_labels)                                     # Trains the classifier on the feature vectors and labels

    os.makedirs("model", exist_ok=True)                      # Creates the 'model' directory if it doesn't exist
    joblib.dump(vectorizer, "model/vectorizer.joblib")       # Saves the trained vectorizer to disk
    joblib.dump(clf, "model/classifier.joblib")              # Saves the trained classifier to disk

    return clf, vectorizer                                   # Returns the trained classifier and vectorizer