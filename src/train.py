from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB         
import joblib                                                # aving Python objects to disk
import os                                             

def train_model(X_texts, y_labels):                    
    vectorizer = CountVectorizer()                           # CountVectorizer instance (bag-of-words) 
    # vectorizer: count number of times each word appears, and creates a table of word counts <- feature vector
    X = vectorizer.fit_transform(X_texts)                    # Learns the vocabulary and transforms texts to feature vectors

    clf = MultinomialNB()                              
    # classifier: learns the probabilities of each word given a class to predict sentiment (positive/negative)
    clf.fit(X, y_labels)                                     # Trains the classifier on the feature vectors and labels

    os.makedirs("model", exist_ok=True)                      # Creates the 'model' directory if it doesn't exist
    # saves the trained vectorizer and classifier to disk
    joblib.dump(vectorizer, "model/vectorizer.joblib")      
    joblib.dump(clf, "model/classifier.joblib")     

    return clf, vectorizer                                 