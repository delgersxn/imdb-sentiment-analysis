from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from config import tfidf_max_features

# TODO: Could add more preprocessing (for example: remove stop)


def build_sklearn_nb():
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=tfidf_max_features)),
        ('nb',    MultinomialNB())
    ])
