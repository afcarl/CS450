# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 21:24:15 2016

@author: nick
"""

import sys
import random
import collections
from sklearn import datasets

# 5. Create a class for a "HardCoded" classifier
class HardCoded():
    guess = "something"    
    
    def train(self, dataset):
        for item in dataset:
            item # pretend like we are doing something
            
        # based on training, guess a zero
        self.guess = 0
        return
        
    def predict(self, array):
        predictions = []
        for item in array:
            predictions.append(self.guess)
        return predictions    

def testClassifier():
    # 1. Load a dataset containing many instances each with a set of attributes 
    #    and a target value.
    # 2. use the popular Iris dataset, found at: 
    #    http://archive.ics.uci.edu/ml/datasets/Iris
    from sklearn import datasets
    iris = datasets.load_iris() 
    
    # Show the data (the attributes of each instance)
    print("iris data:")
    print(iris.data)
    
    # Show the target values (in numeric format) of each instance
    print("iris target:")
    print(iris.target)
    
    # Show the actual target names that correspond to each number
    print("target names")
    print(iris.target_names)
    
    # 3. Randomize the order of the instances in the dataset. Don't forget that 
    #    you need to keep the targets matched up with the appropriate instance.
    data = list(zip(iris.data, iris.target))
    random.shuffle(data)
    iris.data, iris.target = zip(*data)
    
    # 4. Split the dataset into two sets: a training set (70%) and a testing 
    #    set (30%).
    train = {"data": [], "target": []}
    test = {"data": [], "target": []}
    for i in range(0, len(iris.data)):
        if i < (len(iris.data) * .7):
            train["data"].append(iris.data[i])
            train["target"].append(iris.data[i])
        else:
            test["data"].append(iris.data[i])
            test["target"].append(iris.target[i])
            
    print("training set")
    print(train)
    print("test set")
    print(test)
    
    # 6. Instantiate your new classifier, "train" it with your training data, 
    #    then use it to make predictions on the test data.
    classifier = HardCoded()
    print()
    classifier.train(train)
    
    # 7. Determine the accuracy of your classifier's predictions and report the 
    #    result as a percentage.
    predictions = classifier.predict(test["data"])
    count = 0
    for target, prediction in zip(test["target"], predictions):
        if target == prediction:
            count += 1
            
    print("accuracy: %s%%" % round((count / len(test["target"]) * 100), 2))
    return

def main(argv):
    # check args here
    testClassifier()
            
    
#    testClassifier()
    return

if __name__ == "__main__":
    main(sys.argv)