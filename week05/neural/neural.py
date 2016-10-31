import numpy as np
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, Normalizer
# from sklearn.neural_network import MLPClassifier
import time, argparse, sys


class NeuronLayer:
    def __init__(self, num_inputs, num_nodes):
        self.weights = 2 * np.random.random((num_inputs, num_nodes)) - 1
        self.activations = []
        self.errors = []
        self.layer_number = 0  # maybe?


class NeuralNetwork:
    def __init__(self, input_vectors, targets, num_hidden=0, out_type='sigmoid', verbosity=0):
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
        # self.input_vectors = np.concatenate((input_vectors, -np.ones((len(input_vectors), 1))), axis=1)
        self.input_vectors = input_vectors
        self.ndata = len(self.input_vectors)

        # the number of columns/input nodes/attributes
        self.nin = len(self.input_vectors[0])

        # determined by the number of unique target values
        self.nout = len(set(targets))

        # array of validation targets
        self.targets = targets
        self.array_targets = self.set_targets(self.targets)

        # holds the networks output (sequential updating)
        self.output_values = np.zeros(self.nout)

        # set up network
        self.nhidden = num_hidden

        # this makes up the network
        self.hidden_layers = []
        self.output_layer = 0

        # if SLP
        if self.nhidden is 0:
            # print('SLP')
            self.output_layer = NeuronLayer(num_inputs=self.nin, num_nodes=self.nout)

        # one hidden layer of nhidden nodes
        elif type(num_hidden) is not tuple:
            # print('MLP')
            # print('one hidden layer')
            # weights for first layer: determined by number of input vectors and number of hidden nodes
            # weights for second layer: determined by number of inputs from previous layer and number of output nodes

            # self.nhidden += 1  # for the bias node

            # first layer, the hidden layer
            self.hidden_layers.append(NeuronLayer(num_inputs=self.nin + 1, num_nodes=self.nhidden))

            # second layer, the output layer
            self.output_layer = NeuronLayer(num_inputs=self.nhidden + 1, num_nodes=self.nout)

        # multiple hidden hidden_layers
        # weight dimensions are determined by how many inputs are coming from the previous layer, whether its the
        # input vector or a previous hidden layer, and the number of nodes in the layer, which is determined by the
        # num_layers variable
        else:
            # print('multiple hidden hidden_layers')

            for layer, nodes in enumerate(self.nhidden):
                self.hidden_layers.append(
                    NeuronLayer(
                        num_inputs=self.nin + 1 if layer is 0 else self.nhidden[layer - 1] + 1,
                        num_nodes=nodes
                    )
                )

            # last layer, the output layer
            self.output_layer = NeuronLayer(num_inputs=self.nhidden[-1] + 1, num_nodes=self.nout)

        # specify the activation method, default is sigmoid function 1/(1+e^activations)
        self.out_type = out_type

        # calc_accuracy data
        self.the_accuracy = 0

        self.verbosity = verbosity

    def set_targets(self, the_targets):
        """
        Returns an array of targets that can be easily used to calculate error when we have the network outputs
        Hint: LabelEncoder might be the best thing for this function to work
        :return:
        """
        targets = np.zeros((
            len(the_targets),
            self.nout
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
        print('\n# training network output: ')

        file_name = time.strftime("%d-%m-%Y-%H-%M-%S-training.csv")
        accuracy_file = open('/Users/nick/Dropbox/nicknelson/school/BYUI/2016/cs450/week05/accuracy/' +
                             file_name, 'a')

        highest_accuracy = 0
        highest = ''

        self.learn_rate = learn_rate

        self.input_vectors = np.concatenate((self.input_vectors, -np.ones((len(self.input_vectors), 1))), axis=1)

        # print('iterations: ' + str(num_iterations))

        # for each iteration
        for it in range(num_iterations):
            # forward phase
            self.forward()

            # backward phase
            self.backward()

            # csv format: iteration,calc_accuracy
            accuracy = round(self.calc_accuracy(), 6)
            data = str(it) + ',' + str(accuracy)
            accuracy_file.writelines(data + '\n')

            if (self.verbosity is not 0) and (it % self.verbosity == 0):
                print(data)

            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                highest = str(it) + ', ' + str(round(accuracy, 2)) + '%'

        print('\n# Data file: ' + file_name)

        self.the_accuracy = accuracy
        print('\n# Highest accuracy (iteration, accuracy): ' + highest)

    def forward(self):
        """
        This is the forward section of the algorithm
        :param input_vectors:
        :return:
        """
        # compute the activation of each neuron in the hidden layers
        for a_layer, the_layer in enumerate(self.hidden_layers):
            self.hidden_layers[a_layer].activations = 1.0 / (
                1.0 + np.exp(
                    -np.dot(  # this does all input vectors at once
                        self.input_vectors if a_layer is 0 else self.hidden_layers[a_layer - 1].activations,
                        the_layer.weights
                    )
                )
            )
            self.hidden_layers[a_layer].activations = np.concatenate((  # this adds the bias node for this layer
                self.hidden_layers[a_layer].activations,
                -np.ones((len(self.hidden_layers[a_layer].activations), 1))),
                axis=1
            )

        # work through the network until you get to the output hidden_layers neurons
        self.output_layer.activations = 1.0 / (
            1.0 + np.exp(-np.dot(self.hidden_layers[-1].activations, self.output_layer.weights))
        )

        self.output_values = self.output_layer.activations

    def backward(self):
        """
        This is the backward section of the algorithm
        :param input_vectors:
        :param targets:
        :return:
        """
        # compute the error at the output
        self.output_layer.errors = self.output_layer.activations * (
            1.0 - self.output_layer.activations
        ) * (
            self.output_layer.activations - self.array_targets  # the example code in the book reverses these two...
        )
        # print('output error')
        # print(self.output_layer.errors)

        # compute the error in the hidden layer(s)
        # last hidden layer that connects to the output layer
        self.hidden_layers[-1].errors = self.hidden_layers[-1].activations * (
            1.0 - self.hidden_layers[-1].activations
        ) * np.dot(
            self.output_layer.errors, np.transpose(self.output_layer.weights)
        )
        self.hidden_layers[-1].errors = self.hidden_layers[-1].errors[:, :-1]

        # all the other hidden layers
        for a_layer, the_layer in reversed(list(enumerate(self.hidden_layers))):
            if a_layer is len(self.hidden_layers) - 1:
                continue
            self.hidden_layers[a_layer].errors = self.hidden_layers[a_layer].activations * (
                1 - self.hidden_layers[a_layer].activations
            ) * np.dot(
                self.hidden_layers[a_layer + 1].errors,
                np.transpose(self.hidden_layers[a_layer + 1].weights)
            )
            self.hidden_layers[a_layer].errors = self.hidden_layers[a_layer].errors[:, :-1]

        # update the output layer weights
        self.output_layer.weights -= self.learn_rate * np.dot(
            np.transpose(self.hidden_layers[-1].activations),
            self.output_layer.errors
        )

        # update the hidden layer weights
        for layer_number, _ in reversed(list(enumerate(self.hidden_layers))):
            product = np.dot(
                np.transpose(self.hidden_layers[layer_number - 1].activations if layer_number is not 0 else self.input_vectors),
                self.hidden_layers[layer_number].errors
            )
            update = self.learn_rate * product
            self.hidden_layers[layer_number].weights -= update

    def validate(self, X, y, nit):
        """
        I think this function is used for running inputs that don't have targets
        :param input_vectors:
        :return:
        """
        # accuracy_file = open('/Users/nick/Dropbox/nicknelson/school/BYUI/2016/cs450/week05/accuracy/' +
        #                      time.strftime("%d-%m-%Y-%H-%M-%S-training") + '.csv', 'a')

        self.input_vectors = np.concatenate((X, -np.ones((len(X), 1))), axis=1)
        self.targets = y
        self.array_targets = self.set_targets(self.targets)

        # for it in range(nit):
        self.forward()

        accuracy = str(self.calc_accuracy())
        # data = str(it) + ',' + accuracy'
        print('\n# Accuracy: ')
        print(accuracy)
        # accuracy_file.writelines(accuracy + '\n')


    def calc_accuracy(self):
        """
        Return a precentage of how many times we were right vs how many times we were wrong
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

        return correct.count(True) / len(correct) * 100

    def get_accuracy(self):
        return self.the_accuracy


def load_data(which_data, preprocess_method):
    """
    This function handles data retrieval and normalization
    :param which_data:
    :return:
    """
    file_orig = open(
        '/Users/nick/Dropbox/nicknelson/school/BYUI/2016/cs450/week05/data_sets/' +
        which_data + '-orig.csv',
        'w'
    )
    file_preproc = open(
        '/Users/nick/Dropbox/nicknelson/school/BYUI/2016/cs450/week05/data_sets/' +
        which_data + '-preproc.csv',
        'w'
    )
    ds_orig = ''
    ds_preproc = ''

    if which_data == 'and':
        ds_orig = pd.DataFrame([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
        ds_preproc = ds_orig.copy()
    elif which_data == 'xor':
        ds_orig = pd.DataFrame([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
        ds_preproc = ds_orig.copy()
    elif which_data == 'iris':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            header=None
        )
        ds_preproc = ds_orig.copy()
        ds_preproc[4] = LabelEncoder().fit_transform(ds_orig[4])
        # ds_orig = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(ds_orig).toarray())

    elif which_data == 'pima':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
        )

        # the book suggested doing this for the pima dataset, not sure what else to do for this data set
        pima = ds_orig.as_matrix()
        pima[np.where(pima[:, 0] > 8), 0] = 8
        pima[np.where((pima[:, 7] > 20) & (pima[:, 7] <= 30)), 7] = 1
        pima[np.where((pima[:, 7] > 30) & (pima[:, 7] <= 40)), 7] = 2
        pima[np.where((pima[:, 7] > 40) & (pima[:, 7] <= 50)), 7] = 3
        pima[np.where((pima[:, 7] > 50) & (pima[:, 7] <= 60)), 7] = 4
        pima[np.where((pima[:, 7] > 60) & (pima[:, 7] <= 70)), 7] = 5
        pima[np.where((pima[:, 7] > 70) & (pima[:, 7] <= 80)), 7] = 6
        pima[np.where((pima[:, 7] > 80) & (pima[:, 7] <= 90)), 7] = 7

        scaler = StandardScaler()
        ds_preproc = pd.DataFrame(scaler.fit_transform(pima))

    elif which_data == 'cars':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
            header=None
        )
        for i in ds_orig:
            ds_orig[i] = LabelEncoder().fit_transform(ds_orig[i])
        ds_orig = pd.DataFrame(OneHotEncoder(dtype=np.int).fit_transform(ds_orig).toarray())

    elif which_data == 'breast_cancer':
        ds_orig = pd.DataFrame(
            OneHotEncoder(dtype=np.int).fit_transform(
                Imputer(missing_values='NaN', strategy='mean', axis=0).fit_transform(
                    pd.read_csv(
                        'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/'
                        + 'breast-cancer-wisconsin.data',
                        header=None
                    ).replace({'?': np.nan}))
            ).toarray()
        )

    elif which_data == 'la_stop':
        ds_orig = pd.read_csv(
            '/Users/nick/Downloads/Stop_Data_Open_Data-2015.csv'
        )

    elif which_data == 'lenses':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/lenses/lenses.data',
            header=None
        )

    elif which_data == 'voting':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'
        )

    elif which_data == 'credit':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data'
        )

    elif which_data == 'chess':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data'
        )

    elif which_data == 'adults' or which_data == 'adult':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
            names=[
                "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status", "Occupation",
                "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"],
            sep=r'\s*,\s*',
            engine='python',
            na_values="?"
        )
        ds_preproc = ds_orig.copy()
        encoder = LabelEncoder()
        encoder.fit(ds_preproc)
        ds_preproc = encoder.transform(ds_preproc)

    elif which_data == 'bank':
        ds_orig = pd.read_csv(
            '/Users/nick/Dropbox/nicknelson/school/BYUI/2016/cs450/week05/data_sets/bank-full-orig.csv',
        )
        # print(ds_orig)
        bank = ds_orig.as_matrix()

        for i, age in enumerate(range(0, 100, 10)):
            bank[np.where((bank[:, 0] > age) & (bank[:, 0] <= age + 10)), 0] = i

        for i in range(17):
            if i not in [0, 5, 9, 11, 12, 13, 14]:
                bank[:, i] = LabelEncoder().fit_transform(bank[:, i])

        # bank_targets = bank[:, -1:]

        scaler = StandardScaler()
        # bank_scaled = scaler.fit_transform(bank[:, :-1])
        # bank = np.concatenate((bank_scaled, bank_targets), axis=1)

        ds_preproc = pd.DataFrame(np.array(np.concatenate((scaler.fit_transform(bank[:, :-1]), bank[:, -1:]), axis=1), dtype=float))

    elif which_data == 'poker':
        ds_orig = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data'
        )
        with pd.option_context('display.max_rows', len(ds_orig)):
            file_orig.writelines(ds_orig.to_csv(index=False, header=False))

    else:
        print('No data found for \'' + which_data + '\'')
        return None

    # # this makes the neural network work way better
    # if 'Normalize' in preprocess_method:
    #     scaler = Normalizer().fit(ds_preproc)
    #     ds_preproc = scaler.transform(ds_preproc)
    # elif 'StandardScaler' in preprocess_method:
    #     scaler = StandardScaler()
    #     ds_preproc = pd.DataFrame(scaler.fit_transform(pd.DataFrame(ds_preproc.as_matrix()[:, :-1])))
    # elif 'MinMaxScaler' in preprocess_method:
    #     scaler = MinMaxScaler(feature_range=(0, 1))
    #     ds_preproc = pd.DataFrame(scaler.fit_transform(ds_preproc))

    # lets save the data just so we can look at it
    with pd.option_context('display.max_rows', len(ds_orig)):
        file_orig.writelines(ds_orig.to_csv(index=False, header=False))

    with pd.option_context('display.max_rows', len(ds_orig)):
        file_preproc.writelines(ds_preproc.to_csv(index=False, header=False))

    # print(ds_preproc)
    return ds_preproc


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


# TODO change this function to just run the data with an existing neural network implementation
# def process_data(data):
#     # split data
#     train_data, train_target, test_data, test_target = split_data(data, 0.7)
#
#     # print(data)
#     # print(train_data)
#     # print(train_target)
#     # print(test_data)
#     # print(test_target)
#
#     # existing classifier
#
#     # my implementation
#     my_perceptron = NeuralNetwork(train_data, train_target, len(set(np.concatenate((train_target, test_target)))))
#     my_mlp = NeuralNetwork(input_vectors=train_data, targets=train_target, num_neurons=3, num_layers=(5, 5, 5))
#     print('# activations: ')
#     print(my_perceptron.calc_outputs(0))

# def load_pima(split_ration):
#     column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#     ds_orig = pd.read_csv(
#         'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
#         names=column_names
#     )



def main(argv):
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-d', '--data', help='the data set to use, default is iris', required=False, default='iris')
    parser.add_argument('-i', '--num-iterations',
                        help='the number of iterations to run through the data set, default is 200', required=False,
                        default=200, type=int)
    parser.add_argument('-m', '--nhidden', help='the number of hidden nodes/layers to use in the network, default is 4',
                        required=False, default=(4, ), nargs="+")
    parser.add_argument('-p', '--pre-process', help='the kind of pre-processing to use, default is StandardScaler' +
                        '\nother options are: MinMaxScaler, Normalize',
                        required=False, default='StandardScaler')
    parser.add_argument('-n', '--learn-rate', help='the learn rate of the algorithm, default is 0.1', required=False,
                        default=0.1)
    parser.add_argument('-v', '--verbose', help='output more information during execution', required=False,
                        default=0)

    args = parser.parse_args()
    data = args.data
    nit = args.num_iterations
    nhidden = tuple(map(int, args.nhidden))
    pre = args.pre_process
    n = float(args.learn_rate)
    v = int(args.verbose)

    if (data == 'and') or (data == 'xor'):
        pre = ''

    print('# load ' + data + ' data')
    ds = load_data(data, pre)
    if ds is None:
        return
    else:
        X, y, P, q = split_data(ds, 0.7)

        # if v is not 0:
        #     print('\n# training inputs: ')
        #     print(X)
        #     print('\n# training targets: ')
        #     print(y)

        mlp = NeuralNetwork(input_vectors=X, targets=y, num_hidden=nhidden, verbosity=v)
        mlp.train(learn_rate=n, num_iterations=nit)
        print('\n# Last iteration accuracy: ' + str(nit) + ', %s%%' % round(mlp.get_accuracy(), 2))

        print('\n# Validation:')
        print('\n# Validation inputs:')
        print(P)
        print('\n# Validation targets:')
        print(q)
        mlp.validate(P, q, nit=nit)

        # existing implementation
        # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5, 2), random_state = 1)

    # stdsclr = 'StandardScaler'  # this one works way better for pima, a little better for iris
    # # mmsclr = 'MinMaxScaler'

    # simple test data
    # print('\n# and data: ')
    # and_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # and_targets = np.array([0, 0, 0, 1])
    # # print('# targets: ')
    # # print(and_targets)
    # # and_mlp_mult = NeuralNetwork(input_vectors=and_inputs, targets=and_targets, nhidden=0, out_type='sigmoid')
    # and_mlp_mult = NeuralNetwork(input_vectors=and_inputs, targets=and_targets, num_hidden=2, out_type='sigmoid')
    # and_mlp_mult.train(learn_rate=.1, num_iterations=1000)
    # # and_mlp = NeuralNetwork(input_vectors=and_inputs, targets=and_targets, nhidden=2, out_type='sigmoid')
    # # and_mlp.train(learn_rate=0.1, num_iterations=1)
    # print('# Accuracy: %s%%' % round(and_mlp_mult.calc_accuracy(), 2))

    # print('\n# xor data: ')
    # xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # print(xor_inputs)
    # xor_targets = np.array([0, 1, 1, 0])
    # print('# targets: ')
    # print(xor_targets)
    # xor_mlp = NeuralNetwork(input_vectors=xor_inputs, targets=xor_targets, num_hidden=2, out_type='sigmoid')
    # xor_mlp.train(learn_rate=0.4, num_iterations=10000)
    # print('# Accuracy: %s%%' % round(xor_mlp.calc_accuracy(), 2))

    # load iris data
    # print('\n# iris data')
    # iris_data = load_data('iris', stdsclr)
    # print(iris_data)
    # train_data, train_target, test_data, test_target = split_data(iris_data, 0.7)
    # print('# targets: ')
    # print(train_target)
    # iris_mlp = NeuralNetwork(input_vectors=train_data, targets=train_target, num_hidden=(4, ), verbosity=1)
    # iris_mlp.train(learn_rate=0.1, num_iterations=100)
    # print('# Accuracy: %s%%' % round(iris_mlp.calc_accuracy(), 2))

    # load pima data
    # print('\n# pima data')
    # pima_data = load_data('pima', stdsclr)
    # # pima_data = load_data('pima', mmsclr)
    # # print(pima_data)
    # # print('# targets: ')
    # train_data, train_target, test_data, test_target = split_data(pima_data, 0.7)
    # # train_data, train_target, test_data, test_target = load_pima(0.7)
    # # print(train_target)
    # pima_mlp = NeuralNetwork(input_vectors=train_data, targets=train_target, num_hidden=8)
    # nit = 100000
    # pima_mlp.train(learn_rate=0.1, num_iterations=nit)
    # print('# Last iteration calc_accuracy: ' + str(nit) + ', %s%%' % round(pima_mlp.calc_accuracy(), 2))

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