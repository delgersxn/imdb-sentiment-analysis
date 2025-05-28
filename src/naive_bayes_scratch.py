import math


class build_scratch_nb:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}
        self.word_prob = {}
        self.vocab = set()
        self.total_words_per_class_for_unseen = {}

    def train(self, messages, labels):
        processed = []
        for msg in messages:
            words = msg.lower().split()
            processed.append(words)
            self.vocab.update(words)

        num_messages = len(processed)
        unique_words_count = len(self.vocab)

        class_counts = {}
        word_counts_in_class = {}
        total_words_in_class = {}

        for words, label in zip(processed, labels):
            class_counts[label] = class_counts.get(label, 0) + 1
            total_words_in_class[label] = total_words_in_class.get(
                label, 0) + len(words)
            word_counts_in_class.setdefault(label, {})
            for w in words:
                word_counts_in_class[label][w] = word_counts_in_class[label].get(
                    w, 0) + 1

        for label in class_counts:
            self.class_priors[label] = class_counts[label] / num_messages
            self.word_prob[label] = {}
            self.total_words_per_class_for_unseen[label] = total_words_in_class[label]
            denom = total_words_in_class[label] + \
                self.alpha * unique_words_count

            for w in self.vocab:
                numer = word_counts_in_class[label].get(w, 0) + self.alpha
                self.word_prob[label][w] = numer / denom

    def fit(self, messages, labels):
        self.train(messages, labels)
        return self

    def predict(self, messages):
        preds = []
        for msg in messages:
            words = msg.lower().split()
            scores = {}
            total_vocab = len(self.vocab)
            for label, prior in self.class_priors.items():
                score = math.log(prior)
                total_unseen = self.total_words_per_class_for_unseen[label]
                for w in words:
                    if w in self.vocab:
                        score += math.log(self.word_prob[label][w])
                    else:
                        num = self.alpha
                        den = total_unseen + self.alpha * total_vocab
                        score += math.log(num / den)
                scores[label] = score
            preds.append(max(scores, key=scores.get))
        return preds

    def score(self, messages, labels):
        preds = self.predict(messages)
        correct = sum(p == y for p, y in zip(preds, labels))
        return correct / len(labels)
