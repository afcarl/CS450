# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 21:24:15 2016

@author: nick
"""

import sys, random
import pandas as pd
import numpy as np
from sklearn import datasets, preprocessing
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split


class KNN():
    # dataClass =

    # page 159 in the book
    def knn(self, k, data, dataclass, inputs):
        ninputs = np.shape(inputs)[0]
        closest = np.zeros(ninputs)

        for n in range(ninputs):
            # compute distances
            distances = np.sum((data-inputs[n,:])**2, axis=1)

            # identify the nearest neighbors
            indices = np.argsort(distances, axis=0)

            classes = np.unique(dataclass[indices[:k]])
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)
                for i in range(k):
                    counts[dataclass[indices[i]]] += 1
                closest[n] = np.max(counts)

        return closest

    # classify
    def classify(self, k, dataset):


        return


def load_data(which_data):
    data_dir = '/Users/nick/Dropbox/nicknelson/school/BYUI/2016/cs450/week01/'
    data_set = ''

    if which_data == 'iris':
        data_set = pd.read_csv(data_dir + 'iris.data', header=None)
        data_set[4] = preprocessing.LabelEncoder().fit_transform(data_set[4])

        # data_set = pd.io.parsers.read_csv(
        #     'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        #     header=None,
        #     # usecols=[0, 1, 2, 3, 4] # i don't think i need this if i want all the columns
        # )
        # data_set = datasets.load_iris()
        # data_set[4] = preprocessing.LabelEncoder().fit_transform(data_set[4])
        # data_set.columns = ['sl', 'sw', 'pl', 'pw', 'class']
        # print(data_set)
    elif which_data == 'cars':
        data_set = pd.io.parsers.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data',
            header=None,
            # usecols=[0, 1, 2, 3, 4] # i don't think i need this if i want all the columns
        )
        data_set.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
    elif which_data == 'breast_cancer':
        data_set = pd.io.parsers.read_csv(
            'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
            header=None,
            # usecols=[0, 1, 2, 3, 4] # i don't think i need this if i want all the columns
        )
        data_set.columns = ['id number ', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                            'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                            'Normal Nucleoli', 'Mitoses', 'class']
        data_set = data_set.replace(to_replace='?', value=np.nan)

        imp = Imputer(missing_values='NaN', strategy='mean', axis=0) # replace ? with the mean of all numbers in column
        imp.fit(data_set)
        data_set = imp.transform(data_set)

    else:
        print('No data requested')

    # need to preprocess data
    # http://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features
    # http://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

    # le = preprocessing.LabelEncoder()
    # enc = preprocessing.OneHotEncoder()
    # data_set[4] = le.fit_transform(data_set[4]) # new dataframe with numbers instead of words




    return data_set


# bro burton's code
def split_data(data_set, split_amount, target_index):
    data_set = data_set.as_matrix(columns=[data_set.columns])

    target = np.array([i[-1] for i in data_set])
    data = np.array([np.delete(i, -1) for i in data_set])

    split_index = int(split_amount * len(data))

    indices = np.random.permutation(len(data))

    train_data = data[indices[:split_index]]
    train_target = target[indices[:split_index]]

    test_data = data[indices[split_index:]]
    test_target = target[indices[split_index:]]

    return train_data, train_target, test_data, test_target


def existing(train_data, train_target, test_data, test_target):
    print('existing implementation: sklearn.neighbors KNeighborsClassifier')

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_data, train_target)
    predictions = knn.predict(test_data)

    print(predictions)

    return

def main(argv):
    # check args here
    # testClassifier()

    # load iris data
    print('# load iris data')
    data_set = load_data('iris') # can replace this with cmdline arg
    # split iris data
    train_data, train_target, test_data, test_target = split_data(data_set, 0.7, 4)

    # print('# load car data')
    # data_set = load_data('cars')
    # print(data_set)

    # print('# load breast cancer data')
    # data_set = load_data('breast_cancer')
    # print(data_set)

    # classify
    # <call to my implementation>
    # existing(X_train, X_test, y_train, y_test)
    # existing(train_data, train_target, test_data, test_target)

    return


if __name__ == "__main__":
    main(sys.argv)