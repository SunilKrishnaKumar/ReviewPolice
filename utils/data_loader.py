import pandas as pd

def load_data(filename):
    """
    Parses a data .csv and returns (x, y) data from it.
        x: tuple of (ratings: float[], reviews: object[]) - the columns useful for fake review classification
        y: Numpy array with binary labels for whether a review is fake or not.
     """
    df = pd.read_csv(filename)
    ratings = df['rating'].to_numpy()
    reviews = df['review'].to_numpy()
    y = df['label'].to_numpy()
    return (ratings, reviews), y
