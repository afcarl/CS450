# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 21:24:15 2016

@author: nick
"""

import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder
from sklearn.datasets.base import Bunch


class KNN:
    k = 0           # number of neighbors to return
    # accuracy = 0  # mean accuracy
    # mse = 0       # mean squared error
    # data = 0      #
    X = []          # input array
    y = []          # target array

    def __init__(self, n_neighbors):
        self.k = n_neighbors
        # self.X = 0
        # self.y = 0

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """
        # store the data
        self.X = X
        self.y = y
        return

    def predict(self, X):
        """
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.predict

        :param X: input data
        :return: array of predictions
        """
        # run algorithm
        nInputs = np.shape(X)[0]
        closest = np.zeros(nInputs)

        # print('inputs')
        # print(nInputs)

        for n in range(nInputs):
            # print(n)
            # compute distances
            value = (self.X - X[n, :])**2
            distances = np.sum(value, axis=1)

            # identify the nearest neighbours
            indices = np.argsort(distances, axis=0)
            # print('indices')
            # print(indices)

            classes = np.unique(self.y[indices[:self.k]])
            # print('classes')
            # print(classes)
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                # print('classes:')
                # print(classes)
                counts = np.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.y[indices[i]]] += 1
                closest[n] = np.max(counts)

        return closest

    def score(self, X, y):
        """
        http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier.score

        :param X: input data
        :param y: target data
        :return: float of accuracy
        """
        # first run algorithm
        predictions = self.predict(X)

        # then compute accuracy
        return np.sum(predictions == y) / len(y) * 100

    def error(self, X, y):
        """
        https://www.dataquest.io/blog/k-nearest-neighbors-in-python/

        :param X: input data
        :param y: target data
        :return: float of error
        """
        # first run algorithm
        predictions = self.predict(X)

        # then compute error
        return (((predictions - y) ** 2).sum()) / len(X)


def load_data(which_data):
    data_set = ''

    if which_data == 'iris':

        data_set = pd.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
            header=None
        )
        data_set[4] = LabelEncoder().fit_transform(data_set[4])
        data_set = pd.DataFrame(OneHotEncoder(dtype=np.int)._fit_transform(data_set).toarray())

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

    # sklearn knn classifier
    print('existing implementation: sklearn.neighbors KNeighborsClassifier, k=3')
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data, train_target)
    print('accuracy: %s%%' % round(knn.score(test_data, test_target) * 100, 2))

    # my implementation
    print('my implementation, k=3:')
    knn = KNN(n_neighbors=3)
    knn.fit(train_data, train_target)
    print('accuracy: %s%%' % round(knn.score(test_data, test_target), 2))

def main(argv):
    # load iris data
    print('\n# load iris data')
    data1 = load_data('iris')
    process_data(data1)

    # load cars data
    print('\n# load car data')
    data2 = load_data('cars')
    process_data(data2)

    # load breast cancer data
    print('\n# load breast cancer data')
    data3 = load_data('breast_cancer')
    process_data(data3)


if __name__ == "__main__":
    main(sys.argv)
