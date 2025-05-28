import math # log for better number handling

class SimpleMultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_priors = {}  
        self.word_prob = {} 
        self.vocab = set() 
        self.total_words_per_class_for_unseen = {}

    def train(self, messages, labels):
        # 1) preprocess
        processed_messages = []
        for msg in messages:
            words = msg.lower().split()
            processed_messages.append(words)
            self.vocab.update(words)

        num_messages = len(processed_messages)
        unique_words_count = len(self.vocab)

        # 2) train
        class_counts = {} 
        word_counts_in_class = {}
        total_words_in_class = {} 

        for i in range(num_messages):
            label = labels[i]
            words = processed_messages[i]

            class_counts[label] = class_counts.get(label, 0) + 1

            total_words_in_class[label] = total_words_in_class.get(label, 0) + len(words)
            word_counts_in_class[label] = word_counts_in_class.get(label, {})
            for word in words:
                word_counts_in_class[label][word] = word_counts_in_class[label].get(word, 0) + 1

        # 3) evaluate: calculate Probabilities with Smoothing : classifier
        for label in class_counts:
            self.class_priors[label] = class_counts[label] / num_messages

            self.word_prob[label] = {}
            self.total_words_per_class_for_unseen[label] = total_words_in_class[label]
            denom = total_words_in_class[label] + self.alpha * unique_words_count

            for word in self.vocab:
                numer = word_counts_in_class[label].get(word, 0) + self.alpha
                self.word_prob[label][word] = numer / denom

    def predict(self, new_message):
        processed_message = new_message.lower().split()

        scores = {}
        for label, prior_prob in self.class_priors.items():
            score = math.log(prior_prob)

            for word in processed_message:
                if word in self.vocab: 
                    score += math.log(self.word_prob[label][word])
                else: 
                    unseen_word_numerator = self.alpha 
                    unseen_word_denominator = self.total_words_per_class_for_unseen[label] + self.alpha * len(self.vocab)
                    unseen_word_prob = unseen_word_numerator / unseen_word_denominator
                    score += math.log(unseen_word_prob)
            scores[label] = score

        predicted_class = max(scores, key=scores.get)
        return predicted_class