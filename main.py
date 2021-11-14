import os
from models.review_classifier import ReviewClassifier
from utils.data_loader import load_data

TRAIN_DATA_PATH = os.path.join('datasets', 'train.csv')
TEST_DATA_PATH = os.path.join('datasets', 'dev.csv')

if __name__ == '__main__':
    x_train, y_train = load_data(TRAIN_DATA_PATH)
    x_test, y_test = load_data(TEST_DATA_PATH)

    # Should add saving/loading and conditionally train or test, but this works for now
    model = ReviewClassifier()
    model.train(x_train, y_train)
    accuracy = model.test(x_test, y_test)
    print(f'Accuracy: {accuracy * 100}%')
