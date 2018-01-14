"""
2018-01-14T21:49:15.920Z
-----------------------------------------------
BLASSEL Luc
Data processing, inception model with Keras
-----------------------------------------------
"""
import keras
from keras.datasets import cifar10
import numpy as np
from scipy.misc import imresize
from scipy.misc import imshow
import pickle
import os

import time

file_path = "batches.meta" #file with the labels
resizeFactor = 5 #to resize cifar10 images

def load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.

    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    (taken from cifar10.py, modified by luc blassel)
    """

    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        # In Python 3.X it is important to set the encoding,
        # otherwise an exception is raised here.
        meta = pickle.load(file, encoding='bytes')

    # Load the class-names from the pickled file.
    raw = meta[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names

def getLabelIndexes(labels):
    """
    gets label int from string
    """
    ind = np.zeros((len(labels)),dtype=np.int)
    names = load_class_names()
    c = 0

    for i in range(len(names)):
        if names[i] in labels:
            ind[c] = i
            c += 1
    return ind

def load_data(labels,trainCases,testCases,rnd):
    """
    loads data from cifar10 dataset
    """
    names = load_class_names()
    #initialising arrays for better performance
    sub_x_train = np.zeros((trainCases,resizeFactor*32,resizeFactor*32,3),dtype=np.int)
    sub_y_train = np.zeros((trainCases),dtype=np.int)
    sub_x_test = np.zeros((testCases,resizeFactor*32,resizeFactor*32,3),dtype=np.int)
    sub_y_test = np.zeros((testCases),dtype=np.int)

    # gets int values of wanted labels
    ind = getLabelIndexes(labels)

    #loading existing keras dataset
    (x_train,y_train),(x_test,y_test) = cifar10.load_data()

    c = 0
    for i in range(len(y_train)):
        if y_train[i][0] in ind:
            sub_x_train[c] = imresize(x_train[i],resizeFactor*100,'nearest')
            sub_y_train[c] = y_train[i]
            c += 1
            if c >= trainCases:
                break


    c = 0
    for i in range(len(y_test)):
        if y_test[i][0] in ind:
            sub_x_test[c] = imresize(x_test[i],resizeFactor*100,'nearest')
            sub_y_test[c] = y_test[i]
            c += 1
            if c >= testCases:
                break

    for i in range(5):
        print(names[sub_y_train[i]])
        imshow(sub_x_train[i])

    return sub_x_train, sub_y_train, sub_x_test,sub_y_test

def main():
    wantedLabels = ['dog','truck']
    trainnum = 10
    testnum = 10

    load_data(wantedLabels,trainnum,testnum,False)


if __name__ == "__main__":
    main()
