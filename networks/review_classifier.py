import numpy as np
import tensorflow as tf

class ReviewClassifier(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # TODO: actually make the model
        # should probably run BERT on review text, concatenate review score to hidden state, and run through a classifier

    def call(self, inputs, training=False):
        # inputs: tuple of (ratings, reviews)
        ratings, reviews = inputs

        # Make dummy predictions
        predictions = np.zeros((ratings.shape[0], 2))
        predictions[:, 0] = 1
        return predictions
