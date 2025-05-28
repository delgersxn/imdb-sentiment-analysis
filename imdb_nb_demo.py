import time
from src.ourMultinomialNB import SimpleMultinomialNB
from src.load_data import load_imdb_data  

start_time = time.time()

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

# Print accuracy
accuracy = sum([t == p for t, p in zip(test_df['sentiment'], predictions)]) / len(predictions)
print(f"\nAccuracy: {accuracy:.2f}")

tp = fp = tn = fn = 0
for true, pred in zip(test_df['sentiment'], predictions):
    if true == 1 and pred == 1:
        tp += 1
    elif true == 0 and pred == 0:
        tn += 1
    elif true == 0 and pred == 1:
        fp += 1
    elif true == 1 and pred == 0:
        fn += 1

# for positive class (1)
precision_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if (precision_pos + recall_pos) > 0 else 0
support_pos = sum(t == 1 for t in test_df['sentiment'])

# negative class (0)
precision_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if (precision_neg + recall_neg) > 0 else 0
support_neg = sum(t == 0 for t in test_df['sentiment'])

# macro/weighted (normal) average as both supports are equal
avg_precision = (precision_neg + precision_pos) / 2
avg_recall = (recall_neg + recall_pos) / 2
avg_f1 = (f1_neg + f1_pos) / 2
avg_support = (support_neg + support_pos) / 2 

print("\nClassification Metrics:")
print(f"Negative (0) - Precision: {precision_neg:.2f}, Recall: {recall_neg:.2f}, F1: {f1_neg:.2f}, Support: {support_neg}")
print(f"Positive (1) - Precision: {precision_pos:.2f}, Recall: {recall_pos:.2f}, F1: {f1_pos:.2f}, Support: {support_pos}")
print(f"Average - Precision: {avg_precision:.2f}, Recall: {avg_recall:.2f}, F1: {avg_f1:.2f}, Support: {avg_support:.0f}")

print("\nConfusion Matrix:")
print("               Predicted")
print("           |   0   |   1   |")
print("           ---------------")
print(f"True   0   | {tn:5} | {fp:5} |")
print(f"       1   | {fn:5} | {tp:5} |")

end_time = time.time()
minutes, seconds = divmod(end_time - start_time, 60)
print(f"\nTotal time: {int(minutes)} minutes {seconds:.2f} seconds")