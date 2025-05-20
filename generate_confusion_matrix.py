import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix


def load_imdb_data(data_dir):
    data = {"review": [], "sentiment": []}
    for sentiment in ["pos", "neg"]:
        path = os.path.join(data_dir, sentiment)
        label = 1 if sentiment == "pos" else 0
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as f:
                data["review"].append(f.read())
                data["sentiment"].append(label)
    return pd.DataFrame(data)


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [stemmer.stem(t) for t in tokens]
    return " ".join(tokens)


test_df = load_imdb_data("data/aclImdb/test")
test_df['review'] = test_df['review'].apply(preprocess_text)

vectorizer = joblib.load("model/vectorizer.joblib")
clf = joblib.load("model/classifier.joblib")

X_test = vectorizer.transform(test_df['review'])
y_pred = clf.predict(X_test)

cm = confusion_matrix(test_df['sentiment'], y_pred)
labels = ['Negative', 'Positive']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for IMDb Sentiment Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
