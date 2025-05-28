from nltk.corpus import stopwords               
from nltk.stem import PorterStemmer           
from nltk.tokenize import word_tokenize         

stop_words = set(stopwords.words('english'))    
stemmer = PorterStemmer()                     

def preprocess_text(text):                      
    tokens = word_tokenize(text.lower())         # Converts text to lowercase and tokenizes into words 
    tokens = [t for t in tokens if t.isalpha()]  # Keeps only alphabetic tokens (removes punctuation/numbers)
    tokens = [t for t in tokens if t not in stop_words] # Removes stopwords from tokens (only keeps meaningful words)
    tokens = [stemmer.stem(t) for t in tokens]   # Applies stemming to each token (reduces words to their root form)
    return " ".join(tokens)                      # Joins tokens back into a single string and returns it