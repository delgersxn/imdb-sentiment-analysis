from config import tfidf_max_features, batch_size, epochs, DATA_DIR
from data_preprocessing import vectorize_text
from naive_bayes_sklearn import build_sklearn_nb
from naive_bayes_scratch import build_scratch_nb
from keras_model import build_keras_model
from utils import load_acl_imdb
from evaluate import report_and_plot
import numpy as np

if __name__ == '__main__':
    print("Loading IMDB data ..")
    X_train, y_train, X_test, y_test = load_acl_imdb(DATA_DIR)

    # 1) sklearn NB
    print("Training scikit-learn MultinomialNB ...")
    sk_model = build_sklearn_nb()
    sk_model.fit(X_train, y_train)
    sk_preds = sk_model.predict(X_test)
    acc1 = sk_model.score(X_test, y_test)
    print(f"scikit-learn NB accuracy: {acc1:.4f}")
    report_and_plot("Scikit-learn NB", y_test, sk_preds)

    # 2) scratch NB
    print("Training Scratch MultinomialNB ...")
    sc_model = build_scratch_nb(alpha=1.0)
    sc_model.fit(X_train, y_train)
    sc_preds = sc_model.predict(X_test)
    acc2 = sc_model.score(X_test, y_test)
    print(f"Scratch NB accuracy:       {acc2:.4f}")
    report_and_plot("Scratch NB", y_test, sc_preds)

    # 3) Keras model
    print("Vectorizing for Keras…")
    X_train_vec, vect = vectorize_text(X_train, tfidf_max_features)
    X_test_vec = vect.transform(X_test)

    X_train_arr = X_train_vec.toarray()
    X_test_arr = X_test_vec.toarray()

    y_train_arr = np.array(y_train, dtype=np.int32)
    y_test_arr = np.array(y_test,  dtype=np.int32)

    print("Training Keras model ...")
    km = build_keras_model(X_train_arr.shape[1])
    km.fit(
        X_train_arr,
        y_train_arr,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )
    keras_probs = km.predict(X_test_arr, verbose=0).flatten()
    keras_preds = (keras_probs >= 0.5).astype(int)
    loss, acc3 = km.evaluate(X_test_arr, y_test_arr, verbose=0)
    print(f"Keras model accuracy:      {acc3:.4f}")
    report_and_plot("Keras Model", y_test_arr, keras_preds)
