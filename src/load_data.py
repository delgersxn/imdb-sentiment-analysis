import os                                        # file operations
import pandas as pd                              # data handling

def load_imdb_data(data_dir):                    
    data = {"review": [], "sentiment": []}       # Initializes empty lists for reviews and sentiments
    for sentiment in ["pos", "neg"]:             # Loops over positive and negative folders
        path = os.path.join(data_dir, sentiment) # Builds the path to the sentiment folder
        label = 1 if sentiment == "pos" else 0   # Sets label: 1 for positive, 0 for negative
        for filename in os.listdir(path):     # Loops through each file
            with open(os.path.join(path, filename), encoding='utf-8') as f: # Opens the file
                data["review"].append(f.read())  # Reads and stores the review text
                data["sentiment"].append(label)  # Stores the sentiment label
    return pd.DataFrame(data)                    # Returns a DataFrame with the data