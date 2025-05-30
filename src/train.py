from config import tfidf_max_features, batch_size, epochs, DATA_DIR
from data_preprocessing import vectorize_text
from sklearn_nb import build_sklearn_nb, preprocess
from scratch_nb import build_scratch_nb
from keras_model import build_keras_model
from utils import load_acl_imdb
from evaluate import report_and_plot
import numpy as np
import time

def print_time(label, start):
    minutes, seconds = divmod(time.time() - start, 60)
    print(f"{label} time : {int(minutes)} minutes {seconds:.2f} seconds")

if __name__ == '__main__':
    total_start = time.time()
    print("Loading IMDB data ..")
    train_data, train_label, test_data, test_label = load_acl_imdb(DATA_DIR)

    # 1) sklearn NB
    print("\nTraining scikit-learn MultinomialNB ...")
    sk_start = time.time()
    sk_model = build_sklearn_nb()
    # train_data_sk = [preprocess(text) for text in train_data]
    # test_data_sk = [preprocess(text) for text in test_data]
    sk_model.fit(train_data, train_label)
    sk_preds = sk_model.predict(test_data)
    print_time("scikit-learn NB", sk_start)
    report_and_plot("Scikit-learn NB", test_label, sk_preds)

    # 2) scratch NB
    sc_start = time.time()
    print("\nTraining Scratch MultinomialNB ...")
    sc_model = build_scratch_nb(alpha=1.0)
    sc_model.fit(train_data, train_label)
    sc_preds = sc_model.predict(test_data)
    print_time("Scratch NB", sc_start)
    report_and_plot("Scratch NB", test_label, sc_preds)
    
    # 3) Keras model
    print("\nVectorizing for Keras…")
    keras_start = time.time()
    X_train_vec, vect = vectorize_text(train_data, tfidf_max_features)
    X_test_vec = vect.transform(test_data)

    X_train_arr = X_train_vec.toarray()
    X_test_arr = X_test_vec.toarray()

    y_train_arr = np.array(train_label, dtype=np.int32)
    y_test_arr = np.array(test_label,  dtype=np.int32)

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
    print_time("Keras model", keras_start)
    report_and_plot("Keras Model", y_test_arr, keras_preds)