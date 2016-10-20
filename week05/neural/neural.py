import sys
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder

class Neuron:
    def __init__(self, num_inputs, num_outputs):
        self.inputs = np.zeros(num_inputs)    # the i value
        self.outputs = np.zeros(num_outputs)  # the j value


class NN:
    def __init__(self, input_vectors, targets, num_neurons=1, num_layers=1):
        # input data, two dim array, rows is input vectors, columns is individual input values
        # we also insert a bias input for each row
        self.input_vectors = np.c_[np.full((len(input_vectors), 1), -1, dtype=float), input_vectors]

        #  another way of doing it
        # bias_nodes = np.zeros(len(input_vectors))
        # bias_nodes.fill(-1)
        # self.input_vectors = np.c_[bias_nodes, input_vectors]

        # the number of columns/input nodes/attributes
        self.num_inputs = len(self.input_vectors[0])

        # number of output neurons, default is one
        self.num_outputs = num_neurons

        # the number of layers in the network, default is one
        self.num_layers = num_layers

        # random synapses with mean 0
        self.synapses = 2 * np.random.random((self.num_inputs, self.num_outputs)) - 1

        # array to hold neuron activations
        self.activations = np.zeros(self.num_outputs)

        # array of validation targets
        self.targets = targets

    def calc_output(self, threshold):
        # compute the activations
        self.activations = np.dot(self.input_vectors, self.synapses)

        # threshold the activations
        return np.where(self.activations > threshold, 1, 0)


def load_data(which_data):
    """
    this function handles data retrieval and normalization
    :param which_data:
    :return:
    """
    data_set = ''

    if which_data == 'iris':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            header=None
        )
        # data_set[4] = LabelEncoder().fit_transform(data_set[4])
        # data_set = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(data_set).toarray())

    elif which_data == 'pima':
        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
        )
        # the book suggested doing this for the pima dataset
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
    mynn = NN(train_data, train_target, len(set(np.concatenate((train_target, test_target)))))
    print('# activations: ')
    print(mynn.calc_output(0))

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