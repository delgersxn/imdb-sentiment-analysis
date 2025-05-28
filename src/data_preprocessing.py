from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_text(corpus, max_features):
    vect = TfidfVectorizer(max_features=max_features)
    X = vect.fit_transform(corpus)
    return X, vect
