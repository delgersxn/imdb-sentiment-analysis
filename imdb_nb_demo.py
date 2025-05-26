import os
import time
import pandas as pd
from src.ourMultinomialNB import SimpleMultinomialNB

start_time = time.time()
# Load IMDb data
def load_imdb_data(data_dir):
    data = {"review": [], "sentiment": []}
    for sentiment in ["pos", "neg"]:
        path = os.path.join(data_dir, sentiment)
        label = "Positive" if sentiment == "pos" else "Negative"
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as f:
                data["review"].append(f.read())
                data["sentiment"].append(label)
    return pd.DataFrame(data)

# Load and shuffle data
train_df = load_imdb_data("data/aclImdb/train")
test_df = load_imdb_data("data/aclImdb/test")

# Train your classifier
imdb_classifier = SimpleMultinomialNB()
imdb_classifier.train(list(train_df['review']), list(train_df['sentiment']))

# Predict on the test set
predictions = []
for review in test_df['review']:
    pred = imdb_classifier.predict(review)
    predictions.append(pred)

end_time = time.time()

# Print accuracy
accuracy = sum([t == p for t, p in zip(test_df['sentiment'], predictions)]) / len(predictions)
print(f"\nAccuracy: {accuracy:.2f}")

minutes, seconds = divmod(end_time - start_time, 60)
print(f"\nTotal time: {int(minutes)} minutes {seconds:.2f} seconds")