import numpy as np
import time
import sys
import os
from collections import Counter


class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes=10, learning_rate=0.1, min_count=50, polarity_cutoff=0.01):
        """ Create a Neural Network with the given settings

        :param reviews: input training data with the reviews
        :param labels: input training labels with the truth values
        :param hidden_nodes: number of hidden nodes of the one-layer neural network
        :param learning_rate: learning rate of the trained network
        :param min_count: minimum count of each word to be considered valid
        :param polarity_cutoff: minimum polarity on the positive to negative reviews
        """
        np.random.seed(1)

        # Define attributes
        self.min_count = min_count
        self.polarity_cutoff = polarity_cutoff
        self.pos_neg_ratios = self.calculate_ratios(reviews, labels)

        review_vocab, label_vocab = self.pre_process_data(reviews, labels)
        self.review_vocab = list(review_vocab)
        self.label_vocab = list(label_vocab)
        self.reviews = reviews
        self.labels = labels

        # Word dictionary
        self.word2index = {}

        for ii, word in enumerate(self.review_vocab):
            self.word2index[word] = ii

        # Label dictionary
        self.label2index = {}

        for ii, label in enumerate(self.label_vocab):
            self.label2index[label] = ii

        # Define parameters
        self.input_nodes = len(self.review_vocab)
        self.hidden_nodes = hidden_nodes
        self.output_nodes = 1
        self.learning_rate = learning_rate
        self.weights_0_1, self.weights_1_2, self.layer_1 = self.init_network()

    def pre_process_data(self, reviews, labels):
        """ Processing the data, creating the vocabulary

        :param reviews: Training reviews
        :param labels: Training labels
        :return: The review vocabulary, all the words known that will probably impact the prediction and the
                labels vocabulary.
        """
        review_counter = Counter([word.lower() for review in reviews for word in review.split(" ")])
        review_vocab = set({word for word in review_counter.keys() if review_counter[word] > self.min_count})
        review_vocab = filter(lambda x: np.absolute(self.pos_neg_ratios[x]) > self.polarity_cutoff, review_vocab)

        label_vocab = set({ label.lower() for label in labels })

        return review_vocab, label_vocab

    def calculate_ratios(self):
        """ Calculate the pos_neg_ratio, responsible for calculate the impact of the word into the prediction. To a
        better representative impact, the ratio is calculated by the log of the ratio of the positive impact on the
        negative impact

        :return: the positive to negative log ratio
        """
        pos_counter = Counter()
        neg_counter = Counter()
        pos_neg_ratios = Counter()

        for review, label in zip(self.reviews, self.labels):
            for word in review.split(" "):
                word = word.lower()
                if label == 'POSITIVE':
                    pos_counter[word] += 1
                elif label == 'NEGATIVE':
                    neg_counter[word] += 1

        # Filtering only words that appear more than min_counts
        total_counter = pos_counter + neg_counter
        words = [word for word in total_counter.keys() if total_counter[word] > self.min_count]

        for word in words:
            v = pos_counter[word] / (neg_counter[word] + 1)
            pos_neg_ratios[word] = np.log(v) if v > 1 else -np.log(1/(v+0.01))

        return pos_neg_ratios

    def init_network(self):
        """ Initialize the network

        :return: weights and the input layer
        """
        w_0_1 = np.random.normal(0.0, self.input_nodes**-.5,
                                 (self.input_nodes, self.hidden_nodes))
        w_1_2 = np.random.normal(0.0, self.hidden_nodes**-.5,
                                 (self.hidden_nodes, self.output_nodes))
        layer_1 = np.zeros((1, self.hidden_nodes))

        return w_0_1, w_1_2, layer_1

    @staticmethod
    def get_target_for_label(label):
        """ Converts a label to `0` or `1`

        :param label: The label, either 'POSITIVE' or 'NEGATIVE'
        :return: `0` if the label is `NEGATIVE` and `1` otherwise
        """
        label = label.lower()

        return 1 if label == 'positive' else 0

    @staticmethod
    def sigmoid(x):
        """ Calculate the sigmoid function of x

        :param x: input
        :return: sigmoid value
        """

        return 1 / (1 + np.exp(x))

    @staticmethod
    def sigmoid_derivative(output):
        """ Calculates the sigmoid derivative

        :param output: output of the sigmoid function
        :return: derivative
        """

        return output * (1 - output)

    def train(self):
        """ Training the network
        """

        assert len(self.reviews) == len(self.labels), 'There must be the same amount of training reviews' \
                                                                  'and training labels.'

        training_reviews = list()
        for review in self.reviews:
            review_ids = set([self.word2index[word] for word in review.split(" ") if word in self.word2index.keys()])
            training_reviews.append(list(review_ids))

        correct_so_far = 0
        start = time.time()

        for i in range(len(training_reviews)):
            review, label = training_reviews[i], self.labels[i]
            label = self.get_target_for_label(label)

            # Forward Pass
            self.layer_1 = np.sum(self.weights_0_1[review], axis=0)
            h2 = self.layer_1.dot(self.weights_1_2)
            output = self.sigmoid(h2)

            # Backward Pass
            error = label - output
            output_error = error * self.sigmoid_derivative(output)
            hidden_error = output_error * self.weights_1_2

            delta_w_0_1 = hidden_error
            delta_w_1_2 = self.layer_1.T * output_error

            self.weights_0_1[review] += self.learning_rate * delta_w_0_1
            self.weights_1_2 += self.learning_rate * delta_w_1_2[:, None]

            # Check prediction
            pred = output > 0.5
            correct = pred.squeeze() == label

            correct_so_far += correct

            # Log info
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0
            sys.stdout.write('\rProgress: ' + str(100 * i/float(len(training_reviews)))[:4]
                             + '% Speed(reviews/sec): ' + str(reviews_per_second)[:5]
                             + ' #Correct: ' + str(correct_so_far) + ' Trained: ' + str(i+1)
                             + ' Training Accuracy: ' + str(correct_so_far * 100 / float(i+1))[:4] + '%')

            if i % 2500 == 0:
                print('')

    def run(self, review):
        """ Returns a POSITIVE or NEGATIVE prediction for a given review

        :param review: review to be predicted
        :return: prediction, either POSITIVE or NEGATIVE
        """

        set_review = set()
        for word in review.split(" "):
            word = word.lower()

            if word in self.word2index.keys():
                set_review.add(self.word2index[word])

        review = list(set_review)

        self.layer_1 = np.sum(self.weights_0_1[review], axis=0)
        h2 = self.layer_1.dot(self.weights_1_2)
        output = self.sigmoid(h2)

        return 'POSITIVE' if output[0] >= .5 else 'NEGATIVE'

    def test(self, testing_reviews, testing_labels, verbose=True):
        """ Testing function, to run the trained network into a batch of samples and returns its accuracy

        :param testing_reviews: Testing reviews, list of strings
        :param testing_labels: Testing labels, list of strings
        :param verbose: Whether to print or not the accuracy
        :return: Acc
        """

        correct = 0
        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            correct += 1 if pred == testing_labels[i] else 0

            # Print info
            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            if verbose:
                sys.stdout.write('\rProgress: ' + str(100 * i/float(len(testing_reviews)))[:4]
                                 + '% Speed(reviews/sec): ' + str(reviews_per_second)[:5]
                                 + ' #Correct: ' + str(correct) + ' #Tested: ' + str(i+1)
                                 + ' Testing Accuracy: ' + str(correct * 100 / float(i+1))[:4] + '%')

    def save(self, path):
        """ Save the weights on .npy files

        :param path: Path to the directory which it'll be saved
        """
        assert os.path.isdir(path), 'Path must be a directory!'

        if path[-1] is not '/':
            path += '/'

        print('Saving weights...\n\tWeights_0_1 on ' + path + 'weights_0_1.npy')
        np.save(os.path.join(path, 'weights_0_1.npy'), self.weights_0_1)
        print('\tWeights_1_2 on ' + path + 'weights_0_1.npy')
        np.save(os.path.join(path, 'weights_1_2.npy'), self.weights_1_2)

        print('\nWeights saved successfully!')

    def load(self, path):
        """ Load the weights from .npy files

        :param path: Path to the directory which it'll be loaded
        """
        assert os.path.isdir(path), 'Path must be a directory!'

        if path[-1] is not '/':
            path += '/'

        print('Loading weights...\n\tWeights_0_1 on ' + path + 'weigths_0_1.npy')
        self.weights_0_1 = np.load(os.path.join(path, 'weights_0_1.npy'))
        print('\tWeights_1_2 on ' + path + 'weights_1_2.npy')
        self.weights_1_2 = np.load(os.path.join(path, 'weights_1_2.npy'))

        print('\nWeights loaded successfully!')
