import numpy as np
from networks.review_classifier import ReviewClassifier

class ReviewModel:
    def __init__(self):
        self.network = ReviewClassifier()

    def train(self, x, y, epochs=30):
        ratings, reviews = x
        pass

    def test(self, x, y):
        # Make predictions
        raw_predictions = self.network(x, training=False)
        predictions = np.argmax(raw_predictions, axis=1)

        # Mask predictions to calculate error rates
        positive = y == 1
        num_positive = np.count_nonzero(positive)
        negative = y == 0
        num_negative = np.count_nonzero(negative)
        predicted_negative = predictions == 0
        predicted_positive = predictions == 1

        # Determine true/false positive/negative rate
        # Note that these are based off the actual definitions, not the rates in the dataset
        tp = np.count_nonzero(positive & predicted_positive)
        tn = np.count_nonzero(negative & predicted_negative)
        fp = np.count_nonzero(negative & predicted_positive)
        fn = np.count_nonzero(positive & predicted_negative)
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        fpr = fp / (tn + fp)
        fnr = fn / (tp + fn)

        # Print results
        print(f'Test results:')
        print(f'    True positives: {tp} ({tpr * 100}%)')
        print(f'    - False positives: {fp} ({fpr * 100}%)')
        print(f'    True negatives: {tn} ({tnr * 100}%)')
        print(f'    - False negatives: {fn} ({fnr * 100}%)')
