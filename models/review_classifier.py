import numpy as np
import tensorflow as tf

class ReviewClassifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # TODO: actually make the model
        # should probably run BERT on review text, concatenate review score to hidden state, and run through a classifier

    def _parse_x_dataframe(self, df):
        # Converts a dataframe into a tuple of its columns
        reviews = df['review'].to_numpy()
        ratings = df['rating'].to_numpy()
        return reviews, ratings

    def train(self, x_df, y, epochs=30):
        x = self._parse_x_dataframe(x_df)
        pass

    def test(self, x_df, y):
        # TODO: should probably measure accuracy for real and fake reviews separately since the dataset is unbalanced
        x = self._parse_x_dataframe(x_df)
        raw_predictions = self(x, training=False)
        predictions = np.argmax(raw_predictions, axis=1)
        correct = predictions[predictions == y]
        return len(correct) / len(y)

    def call(self, inputs, training=False):
        # inputs: tuple of (reviews, ratings) returned by _parse_x_dataframe

        reviews, ratings = inputs

        # Make dummy predictions
        predictions = np.zeros((ratings.shape[0], 2))
        predictions[:, 0] = 1
        return predictions
