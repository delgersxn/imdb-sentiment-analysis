from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from config import hidden_units, dropout_rate


def build_keras_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(hidden_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model
