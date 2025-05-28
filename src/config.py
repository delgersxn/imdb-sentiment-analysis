import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Hyperparameters
tfidf_max_features = 10000
batch_size = 128
epochs = 5
hidden_units = 64
dropout_rate = 0.5
