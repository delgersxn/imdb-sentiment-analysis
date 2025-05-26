from nltk.corpus import stopwords                # Imports the list of English stopwords from NLTK
from nltk.stem import PorterStemmer              # Imports the Porter stemming algorithm from NLTK
from nltk.tokenize import word_tokenize          # Imports the word tokenizer from NLTK

stop_words = set(stopwords.words('english'))     # Loads the set of English stopwords
stemmer = PorterStemmer()                        # Creates a PorterStemmer object

def preprocess_text(text):                       # Defines a function to preprocess a single text string
    tokens = word_tokenize(text.lower())         # Converts text to lowercase and tokenizes into words 
    tokens = [t for t in tokens if t.isalpha()]  # Keeps only alphabetic tokens (removes punctuation/numbers)
    tokens = [t for t in tokens if t not in stop_words] # Removes stopwords from tokens (only keeps meaningful words)
    tokens = [stemmer.stem(t) for t in tokens]   # Applies stemming to each token (reduces words to their root form)
    return " ".join(tokens)                      # Joins tokens back into a single string and returns it