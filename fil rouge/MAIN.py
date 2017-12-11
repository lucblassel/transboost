# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2017-11-28 10:51:04
# @Last Modified by:   Romain
# @Last Modified time: 2017-11-30 23:38:58
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


#from cifar10 import num_classes
num_classes = 2

#####################################################################################
#####################################################################################
#							DATA LOADING 											#
#####################################################################################
#####################################################################################
cifar10.maybe_download_and_extract()

class_names_load = cifar10.load_class_names()
class_names_load

images_train, cls_train_load, labels_train_load = cifar10.load_training_data([b'dog',b'truck'])
images_test, cls_test_load, labels_test_load = cifar10.load_test_data([b'dog',b'truck'])
images_test2, cls_test_load2, labels_test_load2 = cifar10.load_test_data([b'ship',b'frog'])
images_test3, cls_test_load3, labels_test_load3 = cifar10.load_test_data([b'deer',b'horse'])


# binarising classes, labels and class names

cls_train,labels_train = class_binariser(cls_train_load),label_binariser(labels_train_load)
cls_test,labels_test = class_binariser(cls_test_load),label_binariser(labels_test_load)
cls_test2,labels_test2 = class_binariser(cls_test_load2),label_binariser(labels_test_load2) # to see if the classifier will work on this other set
cls_test3,labels_test3 = class_binariser(cls_test_load3),label_binariser(labels_test_load3)

class_names = class_name_binariser(class_names_load,cls_test_load)
class_names2 = class_name_binariser(class_names_load,cls_test_load2)
class_names3 = class_name_binariser(class_names_load,cls_test_load3)

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

from IPython.display import Image, display
Image('images/08_transfer_learning_flowchart.png')

def plot_images(class_names,images, cls_true, cls_pred=None, smooth=True):

    cls_true = cls_true[0:9]
    images = images [0:9]

    assert len(images) == len(cls_true)

    # Create figure with sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing.
    if cls_pred is None:
        hspace = 0.3
    else:
        hspace = 0.6
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    for i, ax in enumerate(axes.flat):
        # There may be less than 9 images, ensure it doesn't crash.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i],
                      interpolation=interpolation)

            # Name of the true class.
            cls_true_name = class_names[cls_true[i]]

            # Show true and predicted classes.
            if cls_pred is None:
                xlabel = "True: {0}".format(cls_true_name)
            else:
                # Name of the predicted class.
                cls_pred_name = class_names[cls_pred[i]]

                xlabel = "True: {0}\nPred: {1}".format(cls_true_name, cls_pred_name)

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# Plot the images and labels using our helper-function above.
plot_images(class_names,images=images_test, cls_true=cls_test, smooth=False)
plot_images(class_names2,images=images_test2, cls_true=cls_test2, smooth=False)
plot_images(class_names3,images=images_test3, cls_true=cls_test3, smooth=False)
#####################################################################################
#####################################################################################
#							TRANSFER VALUES CALCULATION								#
#####################################################################################
#####################################################################################

inception.maybe_download()

model = inception.Inception()

from inception import transfer_values_cache

file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar10_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar10_test.pkl')
file_path_cache_test2 = os.path.join(cifar10.data_path, 'inception_cifar10_test2.pkl')
file_path_cache_test3 = os.path.join(cifar10.data_path, 'inception_cifar10_test3.pkl')

print("Processing Inception transfer-values for training-images ...")

##############
# TRAIN SET  #
##############

# Scale images because Inception needs pixels to be between 0 and 255,
# while the CIFAR-10 functions return pixels between 0.0 and 1.0
images_scaled = images_train * 255.0


# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train,
                                              images=images_scaled,
                                              model=model)

##############
# TEST SET   #
##############

print("Processing Inception transfer-values for test-images ...")

# Scale images because Inception needs pixels to be between 0 and 255,
# while the CIFAR-10 functions return pixels between 0.0 and 1.0
images_scaled = images_test * 255.0

# If transfer-values have already been calculated then reload them,
# otherwise calculate them and save them to a cache-file.
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test,
                                             images=images_scaled,
                                             model=model)

##############
# TEST SET 2 #
##############
print("Processing Inception transfer-values for test-images 2 ...")

images_scaled2 = images_test2 * 255.0

transfer_values_test2 = transfer_values_cache(cache_path=file_path_cache_test2,
                                             images=images_scaled2,
                                             model=model)

##############
# TEST SET 3 #
##############
print("Processing Inception transfer-values for test-images 3 ...")

images_scaled3 = images_test3 * 255.0

transfer_values_test3 = transfer_values_cache(cache_path=file_path_cache_test3,
                                             images=images_scaled3,
                                             model=model)

#####################################################################################
#####################################################################################
#			GETTING AN IDEA OF THE SEPARABILTY OF CLASSES							#
#####################################################################################
#####################################################################################

def plot_scatter(values, cls):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()

####################################################
#				ORIGINAL SET					   #										
####################################################

#############
#  HELPERS	#										
#############

def plot_scatter(values, cls):
    # Create a color-map with a different color for each class.
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))

    # Get the color for each sample.
    colors = cmap[cls]

    # Extract the x- and y-values.
    x = values[:, 0]
    y = values[:, 1]

    # Plot it.
    plt.scatter(x, y, color=colors)
    plt.show()

#############
#   PCA 	#										
#############

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
transfer_values = transfer_values_train
cls = cls_train
transfer_values_reduced = pca.fit_transform(transfer_values)
plot_scatter(transfer_values_reduced, cls)


#############
#   t-SNE 	#										
#############

from sklearn.manifold import TSNE
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values)
tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
plot_scatter(transfer_values_reduced, cls)

####################################################
#               OTHER SET                          #                                        
####################################################

#############
#   PCA     #                                       
#############

pca = PCA(n_components=2)
transfer_values = transfer_values_test2
cls = cls_test2
transfer_values_reduced = pca.fit_transform(transfer_values_test2)
plot_scatter(transfer_values_reduced, cls)


#############
#   t-SNE   #                                       
#############

from sklearn.manifold import TSNE
pca = PCA(n_components=50)
transfer_values_50d = pca.fit_transform(transfer_values_test2)
tsne = TSNE(n_components=2)
transfer_values_reduced = tsne.fit_transform(transfer_values_50d)
plot_scatter(transfer_values_reduced, cls)

#####################################################################################
#####################################################################################
#							BUILDING LAST LAYER										#
#####################################################################################
#####################################################################################

####################################################
#				SETTING UP						   #										
####################################################

transfer_len = model.transfer_len

x = tf.placeholder(tf.float32, shape=[None, transfer_len], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')

y_true_cls = tf.argmax(y_true, dimension=1)

# Wrap the transfer-values as a Pretty Tensor object.
x_pretty = pt.wrap(x)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=y_true)


global_step = tf.Variable(initial_value=0,
                          name='global_step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step)

y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()

session.run(tf.global_variables_initializer())

train_batch_size = 64


def random_batch():
    # Number of images (transfer-values) in the training-set.
    num_images = len(transfer_values_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random x and y-values.
    # We use the transfer-values instead of images as x-values.
    x_batch = transfer_values_train[idx]
    y_batch = labels_train[idx]

    return x_batch, y_batch

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(num_iterations):
        # Get a batch of training examples.
        # x_batch now holds a batch of images (transfer-values) and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = random_batch()

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        # We also want to retrieve the global_step counter.
        i_global, _ = session.run([global_step, optimizer],
                                  feed_dict=feed_dict_train)

        # Print status to screen every 100 iterations (and last).
        if (i_global % 100 == 0) or (i == num_iterations - 1):
            # Calculate the accuracy on the training-batch.
            batch_acc = session.run(accuracy,
                                    feed_dict=feed_dict_train)

            # Print status.
            msg = "Global Step: {0:>6}, Training Batch Accuracy: {1:>6.1%}"
            print(msg.format(i_global, batch_acc))

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))



####################################################
def predict_cls_test():
    return predict_cls(transfer_values = transfer_values_test,
                       labels = labels_test,
                       cls_true = cls_test)

def predict_cls_test2():
    return predict_cls(transfer_values = transfer_values_test2,
                       labels = labels_test2,
                       cls_true = cls_test2)

def predict_cls_test3():
    return predict_cls(transfer_values = transfer_values_test3,
                       labels = labels_test3,
                       cls_true = cls_test3)
####################################################



####################################################
#				RUNNING 						   #										
####################################################

optimize(num_iterations=1000)

#############
#  HELPERS	#										
#############

def plot_example_errors(images_set,class_names,cls_true,cls_pred, correct):
    # This function is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # correct is a boolean array whether the predicted class
    # is equal to the true class for each image in the test-set.

    # Negate the boolean array.
    incorrect = (correct == False)
    
    # Get the images from the test-set that have been
    # incorrectly classified.
    images = images_set[incorrect]
    
    # Get the predicted classes for those images.
    cls_pred = cls_pred[incorrect]

    # Get the true classes for those images.
    cls_true = cls_true[incorrect]

    n = min(9, len(images))
    
    # Plot the first n images.
    plot_images(class_names,images=images[0:n],
                cls_true=cls_true[0:n],
                cls_pred=cls_pred[0:n])


# Import a function from sklearn to calculate the confusion-matrix.
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(class_names,cls_pred,cls_true):
    # This is called from print_test_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,  # True class for test-set.
                          y_pred=cls_pred)  # Predicted class.

    # Print the confusion matrix as text.
    for i in range(num_classes):
        # Append the class-name to each line.
        class_name = "({}) {}".format(i, class_names[i])
        print(cm[i, :], class_name)

    # Print the class-numbers for easy reference.
    class_numbers = [" ({0})".format(i) for i in range(num_classes)]
    print("".join(class_numbers))

# Split the data-set in batches of this size to limit RAM usage.
batch_size = 256

def predict_cls(transfer_values, labels, cls_true):
    # Number of images.
    num_images = len(transfer_values)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_images, dtype=np.int)

    # Now calculate the predicted classes for the batches.
    # We will just iterate through all the batches.
    # There might be a more clever and Pythonic way of doing this.

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_images:
        # The ending index for the next batch is denoted j.
        j = min(i + batch_size, num_images)

        # Create a feed-dict with the images and labels
        # between index i and j.
        feed_dict = {x: transfer_values[i:j],
                     y_true: labels[i:j]}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
        
    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    return correct, cls_pred

def classification_accuracy(correct):
    # When averaging a boolean array, False means 0 and True means 1.
    # So we are calculating: number of True / len(correct) which is
    # the same as the classification accuracy.

    # Return the classification accuracy
    # and the number of correct classifications.
    return correct.mean(), correct.sum()

def print_test_accuracy(images,pred_fun,cls_test,class_names,show_example_errors=False,show_confusion_matrix=False):

    # For all the images in the test-set,
    # calculate the predicted classes and whether they are correct.
    #correct, cls_pred = locals()[pred_func_name]()
    correct, cls_pred = pred_fun()
    # Classification accuracy and the number of correct classifications.
    acc, num_correct = classification_accuracy(correct)
    
    # Number of images being classified.
    num_images = len(correct)

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, num_correct, num_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(images,class_names,cls_true=cls_test,cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(class_names=class_names,cls_pred=cls_pred,cls_true=cls_test)

#####################################################################################
#####################################################################################
#							GETTING RESULTS											#
#####################################################################################
#####################################################################################

# ORIGINAL TEST SET
print_test_accuracy(images_test, predict_cls_test,cls_test,class_names,show_example_errors=True,
                    show_confusion_matrix=True)

# OTHER CLASSES TEST SET
print_test_accuracy(images_test2,predict_cls_test2,cls_test2,class_names2,show_example_errors=True,
                    show_confusion_matrix=True)

print_test_accuracy(images_test3,predict_cls_test3,cls_test3,class_names3,show_example_errors=True,
                    show_confusion_matrix=True)