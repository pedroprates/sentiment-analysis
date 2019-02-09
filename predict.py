import argparse
import os
from sentiment_network import SentimentNetwork


def predict(args):
    assert os.path.isdir(args.path), "Path should be a directory"

    network = SentimentNetwork(path=args.path)
    prediction = network.run(args.review)

    print('This review was analyzed: ', args.review)
    print('\n\nThis is a ' + prediction + ' review.')


def parse_arguments():
    parser = argparse.ArgumentParser(description="Prediction of a Sentiment Analysis Network")

    parser.add_argument('review',
                        type=str,
                        help="Review to be predicted")

    parser.add_argument('-p',
                        '--path',
                        default='model',
                        type=str,
                        help="Path to the folder where the model files are")

    return parser.parse_args()


if __name__ == '__main__':
    predict(parse_arguments())
