import sys
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder


class NeuralNetwork:
    def __init__(self, input_vectors, targets, num_neurons=1, num_layers=1):
        """

        :param input_vectors:
        :param targets:
        :param num_neurons: the number of output neurons
        :param num_layers: a tuple, number of values is number of layers, the value is the number of nodes in that layer
        If num_neurons and num_layers are both supplied, the last value of the num_layers tuple will overwrite
        num_neurons (num_outputs).
        """
        # input data, two dim array, rows is input vectors, columns is individual input values
        # we also insert a bias input for each row
        self.input_vectors = np.concatenate((-np.ones((len(input_vectors), 1)), input_vectors), axis=1)

        # another way of doing it
        # bias_nodes = np.zeros(len(input_vectors))
        # bias_nodes.fill(-1)
        # self.input_vectors = np.c_[bias_nodes, input_vectors]

        # yet another way of doing it
        # self.input_vectors = np.c_[np.full((len(input_vectors), 1), -1, dtype=float), input_vectors]

        # the number of columns/input nodes/attributes
        self.num_inputs = len(self.input_vectors[0])

        # number of output neurons, default is one
        self.num_outputs = num_neurons

        # the number of num_layers in the network, default is one
        if type(num_layers) != tuple and num_layers < 2:
            self.num_layers = 1
        elif type(num_layers) == tuple:
            self.num_layers = np.asarray(num_layers)
            self.num_outputs = self.num_layers[-1]

        # random synapses with mean 0, this is the initialisation section of the algorithm
        self.synapses = 2 * np.random.random((self.num_inputs, self.num_outputs)) - 1

        # array to hold neuron activations
        self.activations = np.zeros(self.num_outputs)

        # array of validation targets
        self.targets = targets

    def calc_outputs(self, threshold):
        """
        This is the recall section of the algorithm
        :param threshold:
        :return:
        """
        # compute the activations
        self.activations = np.dot(self.input_vectors, self.synapses)

        # threshold the activations, replace this with sigmoid
        return np.where(self.activations > threshold, 1, 0)

    def train(self, learn_rate, threshold=0, num_iterations=60000):
        """
        This is the train section of the algorithm
        :param learn_rate:
        :param threshold:
        :param num_iterations:
        :return:
        """
        # for each iteration, for each input vector
        for i in range(num_iterations):
            # compute the activation of each neuron
            self.activations = self.calc_outputs(threshold)

            # update each of the weights - ERROR HERE: targets array is different shape than self.activations
            self.synapses -= learn_rate * np.dot(np.transpose(self.input_vectors), self.activations - self.targets)


def load_data(which_data):
    """
    This function handles data retrieval and normalization
    :param which_data:
    :return:
    """
    data_set = ''

    if which_data == 'iris':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            header=None
        )
        data_set[4] = LabelEncoder().fit_transform(data_set[4])
        data_set = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(data_set).toarray())

    elif which_data == 'pima':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
        )
        # the book suggested doing this for the pima dataset, not sure what else to do for this data set
        pima = data_set.as_matrix()
        pima[np.where(pima[:, 0] > 8), 0] = 8
        pima[np.where((pima[:, 7] > 20) & (pima[:, 7] <= 30)), 7] = 1
        pima[np.where((pima[:, 7] > 30) & (pima[:, 7] <= 40)), 7] = 2
        pima[np.where((pima[:, 7] > 40) & (pima[:, 7] <= 50)), 7] = 2
        pima[np.where((pima[:, 7] > 50) & (pima[:, 7] <= 60)), 7] = 2
        pima[np.where((pima[:, 7] > 60) & (pima[:, 7] <= 70)), 7] = 2
        pima[np.where((pima[:, 7] > 70) & (pima[:, 7] <= 80)), 7] = 2
        pima[np.where((pima[:, 7] > 80) & (pima[:, 7] <= 90)), 7] = 2
        data_set = pd.DataFrame(pima)

    elif which_data == 'cars':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
            header=None
        )
        for i in data_set:
            data_set[i] = LabelEncoder().fit_transform(data_set[i])
        data_set = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(data_set).toarray())

    elif which_data == 'breast_cancer':
        data_set = pd.DataFrame(
            OneHotEncoder(dtype=np.int)._fit_transform(
                Imputer(missing_values='NaN', strategy='mean', axis=0).fit_transform(
                    pd.read_csv(
                        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/'
                        + 'breast-cancer-wisconsin.data',
                        header=None
                    ).replace({'?': np.nan}))
            ).toarray()
        )

    elif which_data == 'la_stop':
        data_set = pd.read_csv(
            '/Users/nick/Downloads/Stop_Data_Open_Data-2015.csv'
        )

    elif which_data == 'lenses':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data',
            header=None
        )

    elif which_data == 'voting':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'
        )

    elif which_data == 'credit':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
        )

    elif which_data == 'chess':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data'
        )

    else:
        print('No data requested')

    return data_set

def split_data(data_set, split_amount):
    data = Bunch()
    data.data = data_set.values[:, 0:-1]
    data.target = data_set.values[:, -1]

    split_index = int(split_amount * len(data.data))
    indices = np.random.permutation(len(data.data))
    # indices = range(len(data.data))

    train_data = data.data[indices[:split_index]]
    train_target = data.target[indices[:split_index]]

    test_data = data.data[indices[split_index:]]
    test_target = data.target[indices[split_index:]]

    return train_data, train_target, test_data, test_target


def process_data(data):
    # split data
    train_data, train_target, test_data, test_target = split_data(data, 0.7)

    # print(data)
    # print(train_data)
    # print(train_target)
    # print(test_data)
    # print(test_target)

    # existing classifier

    # my implementation
    # my_perceptron = NeuralNetwork(train_data, train_target, len(set(np.concatenate((train_target, test_target)))))
    my_mlp = NeuralNetwork(input_vectors=train_data, targets=train_target, num_neurons=3, num_layers=(5, 5, 5))
    print('# activations: ')
    # print(my_perceptron.calc_outputs(0))

def main(argv):
    # load iris data
    print('\n# load iris data')
    iris_data = load_data('iris')
    process_data(iris_data)

    # load pima data
    print('\n# load pima data')
    pima_data = load_data('pima')
    process_data(pima_data)

    # load cars data
    # print('\n# load car data')
    # cars_data = load_data('cars')
    # process_data(cars_data)

    # load breast cancer data
    # print('\n# load breast cancer data')
    # breast_cancer_data = load_data('breast_cancer')
    # process_data(breast_cancer_data)

    # load LA stop data
    # print('\n# Load LA stop data')
    # stop_data = load_data('la_stop')
    # print(stop_data)
    # process_data(stop_data)

    # load lenses data
    # print('\n# Load lenses data')
    # lenses_data = load_data('lenses')
    # process_data(lenses_data)

    # print('\n# Load voting data')
    # voting_data = load_data('voting')
    # process_data(voting_data)
    #
    # print('\n# Load credit data')
    # credit_screening_data = load_data('credit')
    # process_data(credit_screening_data)
    #
    # print('\n# Load chess data')
    # chess_data = load_data('chess')
    # process_data(chess_data)


if __name__ == "__main__":
    main(sys.argv)