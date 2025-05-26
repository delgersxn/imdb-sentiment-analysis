import time
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.load_data import load_imdb_data        
from src.preprocess import preprocess_text      
from src.train import train_model          
from src.evaluate import evaluate_model         

start_time = time.time()

print("Loading data...") # training and test into a DataFrame
train_df = load_imdb_data("data/aclImdb/train") 
test_df = load_imdb_data("data/aclImdb/test")  

print("Preprocessing...") # all training and test reviews
train_df['review'] = train_df['review'].apply(preprocess_text)
test_df['review'] = test_df['review'].apply(preprocess_text) 

print("Training model...") # on training data
train_model(train_df['review'], train_df['sentiment'])           

print("Evaluating model...") # on test data
evaluate_model(test_df['review'], test_df['sentiment'])          

end_time = time.time()   
minutes, seconds = divmod(end_time - start_time, 60)
print(f"\nTotal time : {int(minutes)} minutes {seconds:.2f} seconds")


print("Generating confusion matrix...")
clf, vectorizer = train_model(train_df['review'], train_df['sentiment'])
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