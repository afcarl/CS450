import sys
import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder


class NeuronLayer:
    def __init__(self, num_inputs, num_nodes):
        self.weights = 2 * np.random.random((num_inputs, num_nodes)) - 1
        self.activations = np.zeros(num_nodes)
        self.errors = np.zeros(num_nodes)
        self.layer_number = 0  # maybe?


class NeuralNetwork:
    def __init__(self, input_vectors, targets, num_hidden=0, out_type='sigmoid', threshold=0):
        """
        This is the initialize section of the algorithm.
        :param input_vectors:
        :param targets:
        :param num_hidden: a tuple of hidden hidden_layers and the number of nodes in each one, default is zero, which means
        no hidden hidden_layers, so it would be a single layer perceptron.
        :param out_type:
        :param num_neurons:
        :param threshold:
        """
        # input data, two dim array, rows is input vectors, columns is individual input values
        # we also insert a bias input for each row
        self.input_vectors = np.concatenate((input_vectors, -np.ones((len(input_vectors), 1))), axis=1)
        self.num_vectors = len(self.input_vectors)

        # the number of columns/input nodes/attributes
        self.num_inputs = len(self.input_vectors[0])

        # determined by the number of unique target values
        self.num_output_nodes = len(set(targets))

        # array of validation targets
        self.targets = targets
        self.array_targets = self.set_targets(self.targets)

        # holds the networks output (sequential updating)
        self.output_values = np.zeros(self.num_output_nodes)

        # set up network
        self.num_hidden = num_hidden

        # this makes up the network
        self.hidden_layers = []
        self.output_layer = 0

        # if SLP
        if self.num_hidden is 0:
            # print('SLP')
            self.output_layer = NeuronLayer(num_inputs=self.num_inputs, num_nodes=self.num_output_nodes)

        # one hidden layer of num_hidden nodes
        elif type(num_hidden) is not tuple:
            # print('MLP')
            # print('one hidden layer')
            # weights for first layer: determined by number of input vectors and number of hidden nodes
            # weights for second layer: determined by number of inputs from previous layer and number of output nodes

            self.num_hidden += 1  # for the bias node

            # first layer, the hidden layer
            self.hidden_layers.append(NeuronLayer(num_inputs=self.num_inputs, num_nodes=self.num_hidden - 1))

            # second layer, the output layer
            self.output_layer = NeuronLayer(num_inputs=self.num_hidden, num_nodes=self.num_output_nodes)

        # multiple hidden hidden_layers
        # weight dimensions are determined by how many inputs are coming from the previous layer, whether its the
        # input vector or a previous hidden layer, and the number of nodes in the layer, which is determined by the
        # num_layers variable
        else:
            # print('multiple hidden hidden_layers')

            for layer, nodes in enumerate(self.num_hidden):
                self.hidden_layers.append(NeuronLayer(
                    num_inputs=self.num_inputs if layer is 0 else self.num_hidden[layer - 1] + 1,
                    num_nodes=nodes if layer is not len(self.num_hidden) else self.num_output_nodes
                ))

            # last layer, the output layer
            self.output_layer = NeuronLayer(num_inputs=self.num_hidden[-1] + 1, num_nodes=self.num_output_nodes)

        # hold threshold, default is zero
        self.threshold = threshold

        # specify the activation method, default is sigmoid function 1/(1+e^activations)
        self.out_type = out_type

    def set_targets(self, the_targets):
        """
        Returns an array of targets that can be easily used to calculate error when we have the network outputs
        Hint: LabelEncoder might be the best thing for this function to work
        :return:
        """
        targets = np.zeros((
            len(the_targets),
            self.num_output_nodes
        ))

        for target in range(len(the_targets)):
            targets[target][int(the_targets[target])] = 1

        return targets

    def print_network(self):
        # TODO this
        print('finish this')

    def train(self, learn_rate, num_iterations=60000):
        """
        This is the train section of the algorithm
        :param learn_rate:
        :param threshold:
        :param num_iterations:
        :return:
        """

        # for each iteration
        for it in range(num_iterations):
            # for each input vector
            for iv in range(self.num_vectors):
                # forward phase
                self.forward()

                # backward phase
                self.backward()

            self.output_values = self.output_layer.activations
            # print(self.output_values)

        # TODO: graph accuracy after each epoch

    def forward(self):
        """
        This is the forward section of the algorithm
        :param input_vectors:
        :return:
        """
        # compute the activation of each neuron in the hidden hidden_layers
        for a_layer, the_layer in enumerate(self.hidden_layers):
            self.hidden_layers[a_layer].activations = 1.0 / (
                1.0 + np.exp(
                    -np.dot(
                        self.input_vectors if a_layer is 0 else self.hidden_layers[a_layer - 1].activations,
                        the_layer.weights
                    )
                )
            )
            self.hidden_layers[a_layer].activations = np.concatenate(
                (self.hidden_layers[a_layer].activations, -np.ones((len(self.hidden_layers[a_layer].activations), 1))),
                axis=1
            )

        # work through the network until you get to the output hidden_layers neurons, which have activations
        self.output_layer.activations = 1.0 / (
            1.0 + np.exp(-np.dot(self.hidden_layers[-1].activations, self.output_layer.weights))
        )

    def backward(self):
        """
        This is the backward section of the algorithm
        :param input_vectors:
        :param targets:
        :return:
        """
        # compute the error at the output
        #

        # compute the error in the hidden layer(s)
        #

        # update the output layer weights
        #

        # update the hidden layer weights
        #

    def recall(self, input_vectors):
        """
        I think this function is used for running inputs that don't have targets
        :param input_vectors:
        :return:
        """
        return self.forward(input_vectors)

    def accuracy(self):
        """
        Return a precentage of how many times we were right vs the size of the data set
        :return:
        """
        outputs = np.array(
            np.array(
                [[1 if i == row.max() else 0 for i in row] for row in np.array(self.output_values)]
            )
        )

        correct = list(np.array(
            [[True if outputs[r][c] == cd else False for c, cd in enumerate(
                rd
            )] for r, rd in enumerate(
                np.array(self.array_targets)
            )]
        ).flatten())

        return correct.count(True) / len(correct)


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
        # data_set = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(data_set).toarray())

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
    my_perceptron = NeuralNetwork(train_data, train_target, len(set(np.concatenate((train_target, test_target)))))
    my_mlp = NeuralNetwork(input_vectors=train_data, targets=train_target, num_neurons=3, num_layers=(5, 5, 5))
    print('# activations: ')
    print(my_perceptron.calc_outputs(0))


def main(argv):
    # simple test data
    # print('\n# and data: ')
    # and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # and_targets = np.array([0, 0, 0, 1])
    # print('# targets: ')
    # print(and_targets)
    # and_mlp_mult = NeuralNetwork(input_vectors=and_inputs, targets=and_targets, num_hidden=0, out_type='sigmoid')
    # and_mlp_mult = NeuralNetwork(input_vectors=and_inputs, targets=and_targets, num_hidden=(5, 4), out_type='sigmoid')
    # and_mlp_mult.train(learn_rate=.1, num_iterations=10)
    # and_mlp = NeuralNetwork(input_vectors=and_inputs, targets=and_targets, num_hidden=2, out_type='sigmoid')
    # and_mlp.train(learn_rate=0.1, num_iterations=1)
    # print('# Accuracy: %s%%' % round(and_mlp.accuracy() * 100, 2))

    # print('\n# xor data: ')
    # xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # xor_targets = np.array([0, 1, 1, 0])
    # print('# targets: ')
    # print(xor_targets)
    # xor_mlp = NeuralNetwork(input_vectors=xor_inputs, targets=xor_targets, num_hidden=2, out_type='sigmoid')
    # xor_mlp.train(learn_rate=0.1, num_iterations=1)
    # print('# Accuracy: %s%%' % round(xor_mlp.accuracy() * 100, 2))

    # load iris data
    print('\n# iris data')
    iris_data = load_data('iris')
    # print(iris_data)
    train_data, train_target, test_data, test_target = split_data(iris_data, 0.7)
    # print('# targets: ')
    # print(train_target)
    iris_mlp = NeuralNetwork(input_vectors=train_data, targets=train_target, num_hidden=(4, 4))
    iris_mlp.train(learn_rate=0.1, num_iterations=10)
    print('# Accuracy: %s%%' % round(iris_mlp.accuracy() * 100, 2))

    # load pima data
    # print('\n# pima data')
    # pima_data = load_data('pima')
    # print(pima_data)
    # print('# targets: ')
    # train_data, train_target, test_data, test_target = split_data(pima_data, 0.7)
    # print(train_target)
    # pima_mlp = NeuralNetwork(input_vectors=train_data, targets=train_target, num_hidden=4)
    # pima_mlp.train(learn_rate=0.1, num_iterations=1)
    # print('# Accuracy: %s%%' % round(pima_mlp.accuracy() * 100, 2))

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

    # 11 lines of python example https://iamtrask.github.io/2015/07/12/basic-python-network/
    # X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])  # input dataset
    # y = np.array([[0, 1, 1, 0]]).T                              # output dataset
    # syn0 = 2 * np.random.random((3, 4)) - 1                     # initialize weights randomly with mean 0
    # syn1 = 2 * np.random.random((4, 1)) - 1                     # initialize weights randomly with mean 0
    # for j in range(60000):                                      # 60000 iterations
    #     l1 = 1 / (1 + np.exp(-(np.dot(X, syn0))))               # hidden layer activations
    #     l2 = 1 / (1 + np.exp(-(np.dot(l1, syn1))))              # output layer activations
    #     l2_delta = (y - l2) * (l2 * (1 - l2))                   # output layer error
    #     l1_delta = l2_delta.dot(syn1.T) * (l1 * (1 - l1))       # hidden layer error
    #     syn1 += l1.T.dot(l2_delta)                              # update output layer weights
    #     syn0 += X.T.dot(l1_delta)                               # update hidden layer weights


if __name__ == "__main__":
    main(sys.argv)