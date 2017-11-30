# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2017-11-28 10:51:04
# @Last Modified by:   Romain
# @Last Modified time: 2017-11-30 16:21:57
import os
os.chdir("/Users/Romain/Documents/Cours/APT/IODAA/transboost/fil rouge")
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta

from binariser import *

# Functions and classes for loading and using the Inception model.
import inception

# We use Pretty Tensor to define the new classifier.
import prettytensor as pt


import cifar10


from cifar10 import num_classes

cifar10.maybe_download_and_extract()

class_names_load = cifar10.load_class_names()
class_names_load

images_train, cls_train_load, labels_train_load = cifar10.load_training_data()
images_test, cls_test_load, labels_test_load = cifar10.load_test_data()

# binarising the class and labels and class names

cls_train,labels_train = class_binariser(cls_train_load),label_binariser(labels_train_load)
cls_test,labels_test = class_binariser(cls_test_load),label_binariser(labels_test_load)
class_names = class_name_binariser(class_names_load,cls_train_load)

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

from IPython.display import Image, display
Image('images/08_transfer_learning_flowchart.png')



if __name__ == '__main__':
	main()