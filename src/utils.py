import os


def load_acl_imdb(data_dir):
    train_texts, train_labels = [], []
    test_texts,  test_labels = [], []

    base = os.path.join(data_dir, 'aclImdb')

    for split, texts, labels in [
        ('train', train_texts, train_labels),
        ('test',  test_texts,  test_labels)
    ]:
        for label in ['pos', 'neg']:
            folder = os.path.join(base, split, label)
            for fname in os.listdir(folder):
                if fname.endswith('.txt'):
                    path = os.path.join(folder, fname)
                    with open(path, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                    labels.append(1 if label == 'pos' else 0)
    return train_texts, train_labels, test_texts, test_labels
