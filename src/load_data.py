import os
import pandas as pd


def load_imdb_data(data_dir):
    data = {"review": [], "sentiment": []}
    for sentiment in ["pos", "neg"]:
        path = os.path.join(data_dir, sentiment)
        label = 1 if sentiment == "pos" else 0
        for filename in os.listdir(path):
            with open(os.path.join(path, filename), encoding='utf-8') as f:
                data["review"].append(f.read())
                data["sentiment"].append(label)
    return pd.DataFrame(data)
