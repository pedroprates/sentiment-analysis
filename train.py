import os
import argparse
from sentiment_network import SentimentNetwork


def train(args):
    assert os.path.isfile(args.reviews), 'Reviews must be a file'
    assert os.path.isfile(args.labels), 'Labels must be a file'
    assert os.path.isdir(args.output), 'Output path must be a directory'

    reviews = read_file(args.reviews)
    labels = read_file(args.labels)

    network = SentimentNetwork(reviews[:-1000],
                               labels[:-1000],
                               hidden_nodes=args.hidden,
                               min_count=args.minimum,
                               polarity_cutoff=args.polarity,
                               learning_rate=args.lr)
    network.train()
    network.save(args.output)
    print('\n\nTesting:\n')
    network.test(reviews[-1000:], labels[-1000:], verbose=True)


def read_file(filepath):
    with open(filepath, 'r') as f:
        return list(map(lambda x: x[:-1], f.readlines()))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Network")

    parser.add_argument('-r',
                        '--reviews',
                        default='data/reviews.txt',
                        type=str,
                        help='Path to the reviews file, must be a string.')

    parser.add_argument('-l',
                        '--labels',
                        default='data/labels.txt',
                        type=str,
                        help='Path to the labels file, must be a string')

    parser.add_argument('-o',
                        '--output',
                        default='model/',
                        type=str,
                        help='Path to the output directory, must be a string')

    parser.add_argument('--hidden',
                        default=10,
                        type=int,
                        required=False,
                        help="Number of hidden nodes on the Network")

    parser.add_argument('--lr',
                        default=0.01,
                        type=float,
                        required=False,
                        help='Learning Rate for the network training')

    parser.add_argument('--minimum',
                        default=20,
                        type=int,
                        required=False,
                        help='Minimum count of a word to be considered on the vocabulary')

    parser.add_argument('--polarity',
                        default=0.05,
                        type=float,
                        required=False,
                        help="Minimum absolute polarity on the positive to negative ratio to be considered")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_arguments())
