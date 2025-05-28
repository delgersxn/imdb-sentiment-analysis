from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.naive_bayes import MultinomialNB         
import joblib                                               
import os                                             

def train_model(X_texts, y_labels):                    
    vectorizer = CountVectorizer()                          
    # vectorizer: count number of times each word appears, and creates a table of word counts <- feature vector
    X = vectorizer.fit_transform(X_texts)                 

    clf = MultinomialNB()                              
    # classifier: learns the probabilities of each word given a class to predict sentiment (positive/negative)
    clf.fit(X, y_labels)                                    

    os.makedirs("model", exist_ok=True)                      
    # saves the trained vectorizer and classifier to disk
    joblib.dump(vectorizer, "model/vectorizer.joblib")      
    joblib.dump(clf, "model/classifier.joblib")     

    return clf, vectorizer                                 