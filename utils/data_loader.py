import pandas as pd

def load_data(filename):
    """
    Parses a data .csv and returns (x, y) data from it.
        x: Pandas DataFrame containing the useful columns from the dataset for performing classification.
        y: Numpy array with binary labels for whether a review is fake or not.
     """
    df = pd.read_csv(filename)
    x = df[['rating', 'review']]
    y = df['label'].to_numpy()
    return x, y
