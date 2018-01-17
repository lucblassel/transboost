# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-01-15 14:59:20
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-17T14:11:57+01:00

import numpy as np
import time
from binariser import *
from dataProcessing import *
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout
from keras import backend as k

#####################
# LOADING DATA        #
#####################

def loader(trainLabels,testLabels,trainNum,testNum,**kwargs):
    """
    this function loads the datasets from CIFAR 10 with correct output in order to feed inception
    OUTPUT : 32*resizefactor,32*resizefactor
    """

    raw_train,raw_test = loadRawData()
    x_train,y_train = loadTrainingData(raw_train,labels=trainLabels,trainCases=trainNum)
    # x_train,y_train = loadTrainingData(raw_train,trLabels,trNum)
    x_test,y_test = loadTestingData(raw_test,labels=testLabels,testCases=testNum)
    # x_test,y_test = loadTestingData(raw_test,teLabels,teNum)
    y_train_bin,y_test_bin = binarise(y_train),binarise(y_test)

    return x_train, y_train_bin, x_test, y_test_bin
    # return x_train, y_train, x_test, y_test

#####################################
# BUILDING MODEL FOR TWO CLASSES    #
#####################################

def full_model_builder(originalSize,resizeFactor,**kwargs):
    """
    this function builds a model that outputs binary classes
    INPUTS :
    - img_width >=139
    - img_height >=139
    OUTPUTS :
    -full model
    """
    img_width = originalSize*resizeFactor
    img_height = originalSize*resizeFactor

    model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers:
        layer.trainable = False

    #Adding custom Layers
    x = model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    # x = Dropout(.2)(x)
    predictions = Dense(2, activation="softmax")(x)

    # creating the final model
    model_final = Model(input = model.input, output = predictions)

    # compile the model
    model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

    return model_final

#####################################
# TRAINING AND TESTING FULL MODEL    #
#####################################

def full_model_trainer(model,x_train,y_train_bin,x_test,y_test_bin,epochs,**kwargs):
    """
    this function's purpose is to train the full model
    INPUTS : the model to train
    OUPUTS : the model score
    """
    model.fit(x = x_train, y = y_train_bin, batch_size = 10, epochs = epochs)
    score = model.evaluate(x_test, y_test_bin, verbose=1)
    return score


############################################################################
# TRAINING FIRST LAYERS                                                    #
############################################################################

def first_layers_modified_model_builder(model,layerLimit,**kwargs):
    """
    this function changes a model whose first layers are trainable with reinitialized weights
    INPUTS :
    - model to modifiy
    - layerLimit : limit of the first layer to modify (see layer.name)
    OUTPUTS :
    - copy of the modified model
    """
    model_copy =  model
    for layer in model_copy.layers[:layerLimit]:
        layer.trainable = True
        previous_weights = layer.get_weights()
        new_weights = list((10*np.random.random((np.array(previous_weights).shape))))
        layer.set_weights(new_weights)

    for layer in model_copy.layers[layerLimit:]:
        layer.trainable = False
    return model_copy

def first_layers_modified_model_trainer(model,x_train,y_train_bin,epochs,threshold,**kwargs):
    """
    this function trains models from [first_layers_modified_model_builder] function
    """
    model.fit(x = x_train, y = y_train_bin, batch_size = 10, epochs = epochs,callbacks = [callbackBoosting(threshold)])


#######################################################
#                BOOSTING                             #
#######################################################
def take(tab,indexes):
	output = np.zeros(tab.shape)

	c=0
	for i in indexes:
		output[c] = tab[i]
		c+=1
	return output
# def booster(full_model,times,x_train,y_train_bin,epochs,threshold,layerLimit,**kwargs):
def booster(full_model,x_train,y_train_bin,epochs,threshold,layerLimit,times,**kwargs):
	train_length = len(x_train)
	model_list = []
	error_list = []
	alpha_list = []


	if train_length==0:
		raise NameError("length of training set equals 0")

	prob = np.repeat(1/train_length, train_length)
	indexes = list(range(train_length))

	for time in range(times):
		x_train_boost_indexes = np.random.choice(indexes,p=prob,size=train_length,replace=True)
		x_train_boost = take(x_train,x_train_boost_indexes)

		current_model = first_layers_modified_model_builder(full_model,layerLimit)
		error = 0
		while error == 1 or error == 0 :
			current_model = first_layers_modified_model_builder(full_model,layerLimit)
			first_layers_modified_model_trainer(current_model,x_train_boost,y_train_bin,epochs,threshold)
			error = 1 - current_model.evaluate(x_train, y_train_bin, verbose=1)[1]
		alpha = .5*np.log((1-error)/error)

		error_list.append(error)
		model_list.append(current_model)
		alpha_list.append(alpha)

		predicted_prob = current_model.predict(x_train_boost)
		for i in range(train_length):
			if np.where(predicted_prob[i] == predicted_prob[i].max())[0][0] == np.where(y_train_bin[i] == 1)[0][0]:
				prob[i] = prob[i]*np.exp(-alpha)
			else:
				prob[i] = prob[i]*np.exp(alpha)
		prob = prob / np.sum(prob)

	return model_list, error_list, alpha_list

def prediction_boosting(x,model_list, alpha_list):
	n_samples = len(x)
	n_models = len(model_list)
	results = []
	predicted_class_list = []
	c = 0
	for model in model_list:
		print("beginning prediction for model :",c)
		probas = model.predict(x)
		to_append = []
		for proba in probas:
			predicted_class = np.where(proba == proba.max())[0][0]
			if predicted_class == 0:
				predicted_class = -1
			to_append.append(predicted_class)
		predicted_class_list.append(to_append)
		print("ending prediction for model :",c)
		c +=1
	predicted_class_list = np.array(predicted_class_list)
	predicted_class_list.reshape((n_models,n_samples))
	predicted_class_list = np.transpose(predicted_class_list)
	alpha_list = np.array(alpha_list)
	raw_results = np.dot(predicted_class_list,alpha_list)

	for raw_result in raw_results:
		if raw_result >=0:
			results.append([0,1])
		else:
			results.append([1,0])
	return results

def accuracy(y_true,y_pred):
	if isinstance(y_true,np.ndarray):
		y_true = y_true.tolist()
	if isinstance(y_pred,np.ndarray):
		y_pred = y_pred.tolist()
	bool_res = []
	for i in range(len(y_true)):
		bool_res.append(y_true[i] == y_pred[i])
	int_res = list(map(int,bool_res))
	accuracy = np.sum(int_res)/len(y_true)
	return accuracy

def main():
	""" this function stands for testing purposes
	"""
	wantedLabels=['dog','truck']
	trainnum,testnum = 100,100
	x_train, y_train_bin, x_test, y_test_bin = loader(wantedLabels,trainnum,testnum)
	img_width,img_height = 160,160
	full_model = full_model_builder(img_width,img_height)
	layerLimit = 10
	epochs = 1
	threshold = .5
	times = 5
	model_list, error_list, alpha_list = booster(full_model,x_train,y_train_bin,epochs,threshold,layerLimit,times)
	print("model_list ", model_list)
	print("error_list ", error_list)
	print("alpha_list ", alpha_list)

if __name__ == '__main__':
	main()
