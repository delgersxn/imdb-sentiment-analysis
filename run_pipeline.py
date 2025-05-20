from src.load_data import load_imdb_data
from src.preprocess import preprocess_text
from src.train import train_model
from src.evaluate import evaluate_model

print("Loading data...")
train_df = load_imdb_data("data/aclImdb/train")
test_df = load_imdb_data("data/aclImdb/test")

print("Preprocessing...")
train_df['review'] = train_df['review'].apply(preprocess_text)
test_df['review'] = test_df['review'].apply(preprocess_text)

print("Training model...")
train_model(train_df['review'], train_df['sentiment'])

print("Evaluating model...")
evaluate_model(test_df['review'], test_df['sentiment'])
