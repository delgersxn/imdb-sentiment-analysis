import math # math.log for better number handling

class SimpleMultinomialNB:
    def __init__(self, alpha=1.0):
        # alpha = smoothing factor ("tiny extra bit" Bayes adds)
        # alpha=1.0 is Laplace smoothing
        self.alpha = alpha
        self.class_priors = {}  # Stores P(pos) and P(neg)
        self.word_probabilities = {} # Stores P(word | pos) and P(word | neg)
        self.vocabulary = set() # All unique words
        # Store these for use in prediction's unseen_word_prob calculation.
        self.total_words_per_class_for_unseen = {} # hold the sum of word counts for each class.

    def train(self, messages, labels):
        # 1) preprocess
        processed_messages = []
        for msg in messages:
            words = msg.lower().split() # Make lowercase, split by space
            processed_messages.append(words)
            self.vocabulary.update(words) # Add all words to our vocabulary

        num_messages = len(processed_messages)
        unique_words_count = len(self.vocabulary)

        # 2) train
        # Initialize counts for each class.
        class_counts = {}  # E.g., { 'pos': 0, 'neg': 0 }
        word_counts_in_class = {} # E.g., { 'pos': {'word1': 0, ...}, 'neg': {'word1': 0, ...} }
        total_words_in_class = {} # E.g., { 'pos': 0, 'neg': 0 }

        # Go through each message and its label
        for i in range(num_messages):
            label = labels[i]
            words = processed_messages[i]

            # Count messages per class
            class_counts[label] = class_counts.get(label, 0) + 1

            # Count words per class and total words per class
            total_words_in_class[label] = total_words_in_class.get(label, 0) + len(words)
            # Ensure the inner dictionary for words in this class exists
            word_counts_in_class[label] = word_counts_in_class.get(label, {})
            for word in words:
                word_counts_in_class[label][word] = word_counts_in_class[label].get(word, 0) + 1

        # calculate Probabilities with Smoothing
        for label in class_counts:
            # Calculate P(Class) - Class Prior Probability
            self.class_priors[label] = class_counts[label] / num_messages

            # Calculate P(Word | Class) - Word Likelihoods
            self.word_probabilities[label] = {}
            # Store total_words_in_class for use in prediction for unseen words
            self.total_words_per_class_for_unseen[label] = total_words_in_class[label]

            # Denominator for P(Word | Class) including smoothing
            # (total words in class + alpha * number of unique words in vocabulary)
            denominator = total_words_in_class[label] + self.alpha * unique_words_count

            # Calculate probability for each word in our overall vocabulary for this class
            for word in self.vocabulary:
                # Numerator for P(Word | Class) including smoothing
                # (count of word in this class + alpha)
                numerator = word_counts_in_class[label].get(word, 0) + self.alpha
                self.word_probabilities[label][word] = numerator / denominator

    def predict(self, new_message):
        processed_message = new_message.lower().split()

        scores = {}
        # Iterate through each class ('pos', 'neg')
        for label, prior_prob in self.class_priors.items():
            # Start with log of P(Class)
            # log probabilities avoid multiplying many small numbers together:
            # math rule: log(a*b*c) = log(a) + log(b) + log(c)
            score = math.log(prior_prob)

            # Add log probability of each word in new message
            for word in processed_message:
                if word in self.vocabulary: # If word was seen during training
                    score += math.log(self.word_probabilities[label][word])
                else: # If word was NOT seen in training (an "unseen word")
                    # use a smoothed probability for unseen words to avoid log(0) errors.
                    # This calculation uses stored total_words_per_class_for_unseen and the overall vocabulary size.
                    unseen_word_numerator = self.alpha # Only alpha in numerator for unseen words
                    unseen_word_denominator = self.total_words_per_class_for_unseen[label] + self.alpha * len(self.vocabulary)
                    unseen_word_prob = unseen_word_numerator / unseen_word_denominator
                    score += math.log(unseen_word_prob)
            scores[label] = score

        # Find class with highest score ( most likely class)
        predicted_class = max(scores, key=scores.get)
        return predicted_class


#region --- Let's make Bayes learn and guess! ---

# # 1. Provide Bayes' study material (our training data)
# training_messages = [
#     "I love this good day",
#     "This is a good fun game",
#     "I hate this bad game",
#     "This day is bad"
# ]
# training_labels = ["Happy", "Happy", "Sad", "Sad"]

# # 2. Create our Bayes robot (an instance of our classifier) and teach him!
# bayes_robot = SimpleMultinomialNB()
# bayes_robot.train(training_messages, training_labels)

# # 3. Give Bayes new messages to guess their sentiment
# new_message_1 = "This is a bad day"
# new_message_2 = "I love a good game"
# new_message_3 = "Today is fun" # This message includes "Today", which wasn't in training messages.

# print(f"Message: '{new_message_1}' is predicted as: {bayes_robot.predict(new_message_1)}")
# print(f"Message: '{new_message_2}' is predicted as: {bayes_robot.predict(new_message_2)}")
# print(f"Message: '{new_message_3}' is predicted as: {bayes_robot.predict(new_message_3)}")
#endregion