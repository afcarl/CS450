from __future__ import division
from sklearn import datasets
from sklearn import preprocessing
import numpy as np
import pandas as pd
import sys
from hardcoded import HardCodedClassifier
from sklearn.neighbors import KNeighborsClassifier
from knn import BurtonKnnClassifier

def split_data(dataset, split_amount):
    split_index = split_amount * len(dataset.data)

    indices = np.random.permutation(len(dataset.data))

    train_data = dataset.data[indices[:split_index]]
    train_target = dataset.target[indices[:split_index]]

    test_data = dataset.data[indices[split_index:]]
    test_target = dataset.target[indices[split_index:]]

    return (train_data, train_target, test_data, test_target)


def process_data(dataset, classifier):
    # TODO: scale the data

    split_amount = .70;
    train_data, train_target, test_data, test_target = split_data(dataset, split_amount)

    classifier.fit(train_data, train_target)
    predictions = classifier.predict(test_data)

    test_results = predictions == test_target

    correct = test_results.sum()
    total = len(test_results)
    percent = correct / total
    percent *= 100

    print("{} out of {} for {:.2f}% accuracy".format(correct, total, percent))

def loadFile(file_name):
    data_dir = "/Users/sburton/Dropbox/byui/classes/cs450/datasets/"
    data_file = data_dir + file_name

    data = pd.read_csv(data_file)


def main(argv):
    #dataset = loadFile("car.data")
    dataset = datasets.load_iris()

    classifier = HardCodedClassifier()
    #classifier = KNeighborsClassifier(n_neighbors=3)
    #classifier = BurtonKnnClassifier(3)

    process_data(dataset, classifier)



if __name__ == "__main__":
    main(sys.argv)