# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-01-15 14:59:20
# @Last modified by:   Luc Blassel
# @Last modified time: 2018-01-28T17:23:09+01:00
"""
inspired by https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""

#Project files importing
import paramGetter as pg
from binariser import *
from dataProcessing import *
from callbackBoosting import *
from dataLoader import *

#external packages importing
import sys
import gc
import time
import os.path
import numpy as np
import pandas as pd
import copy as cp
import _pickle as pickle
import tensorflow as tf

from keras import backend as k
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.backend.tensorflow_backend import set_session
from pathlib import Path
from itertools import chain
from datetime import datetime

# TensorFlow wizardry
config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of frac of the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.9
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

def getArgs():
	"""
	gets the parameters from configuration file
	"""
	if len(sys.argv)==1:
		print("please specify path of config file...")
		sys.exit()

	path = sys.argv[1]
	return pg.reader(path) #dictionary with relevant parameters

# checks if models directory already exists, and iuf not creates it
def checkDir(dataPath):
	"""
	checks if model's directory already exists, and if not creates it
	#Input: 
	dataPath: The directory of model's path
	#Output: 
	A string : if the path does not exist, it prints ‘creating’; otherwise ‘exists already’
	"""
	if not os.path.exists(dataPath):
		print("creating" ,dataPath, "directory")
		os.makedirs(dataPath)
	else:
		print("directory {} exists already.".format(dataPath))

def printParams(params):
	"""
	Prints parameters from configuration
	#Input:
	Params: parameters from the configuration file
	#Output:
	Print the each parameters.
	"""
	for param in params:
		print(param+" : ",params[param])

models_path = "models"
models_weights_path = "models_weights"

#1st part
classes_source = ['dog','truck']
classes_target = ['deer','horse']
num_of_classes = len(classes_source)

batch_size_source = 10
transformation_ratio = .05
originalSize = 32
resizeFactor = 5

path_to_train = path + "train"
path_to_validation = path + "validation"
path_to_test = path + "test"

path_to_best_top_model = "best_top_model.hdf5"

#TODO automatise counting of these numbers
# trainNum_source = 7950
# valNum_source = 2040
# testNum_source = 2040

trainNum_target = 8010
valNum_target = 1980
testNum_target = 1980
trainNum_source = 8010
valNum_source = 1980
testNum_source = 1980

lr_source = 0.0001
epochs_source = 10

recompute_transfer_values = False
train_top_model = False

#2nd part
layerLimit = 15
epochs_target = 50
lr_target = 0.0001
batch_size_target = 10
threshold = .65
reinitialize_bottom_layers = False
bigNet = False
times = 100

proba_threshold = .5

#####################################
# BUILDING MODEL FOR TWO CLASSES    #
#####################################

def bottom_layers_builder(originalSize,resizeFactor,**kwargs):
	"""
	Builds the bottom layers of model for the domain of source. It uses the Xception model. 
	#Input:
	originalSize: the original size of images
	resizeFactor: value that represents the image resize factor
	#Output:
	Model: the model of bottom layers with Xception
	"""
	img_size = originalSize*resizeFactor

	if k.image_data_format() == 'channels_first':
		input_shape = (3, img_size, img_size)
	else:
		input_shape = (img_size, img_size, 3)

	#model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_size, img_size, 3))
	model = applications.Xception(weights = "imagenet", include_top=False, input_shape = input_shape)

	for layer in model.layers :
		layer.trainable = False
	return model

def create_generators(path_to_train,path_to_validation,classes_source,batch_size_source,originalSize,resizeFactor,transformation_ratio,**kwargs):
	"""
	Creates generators. Adds noises (rotations, zoom, etc.) in the data and makes the binarization for the labels. 
	#Input:
	Path_to_train: the path of train set 
	path_to_validation: the path of validation set
	classes_source: classes to train from domain of source
	batch_size_source: the size of batch of domain source
	originalSize: the original size of images
	resizeFactor: value that represents the image resize factor
	transformation_ratio: the ratio of transformation when rotates, shears and zooms	
	Output:
	Train_generator : the train set after the generation of the data 
	validation_generator: the validation set after the generation of the data
	test_generator: the test set after the generation of the data

	"""
	img_size = originalSize*resizeFactor

	train_datagen = ImageDataGenerator(rescale=1. / 255,
									   rotation_range=transformation_ratio,
									   shear_range=transformation_ratio,
									   zoom_range=transformation_ratio,
									   cval=transformation_ratio,
									   horizontal_flip=True,
									   vertical_flip=True)

	validation_datagen = ImageDataGenerator(rescale=1. / 255)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(path_to_train,target_size=(img_size, img_size),
														batch_size=batch_size_source,
														classes = classes_source,
														class_mode='binary',
														shuffle = False)

	validation_generator = validation_datagen.flow_from_directory(path_to_validation,target_size=(img_size, img_size),
																  classes = classes_source,
																  batch_size=batch_size_source,
																  class_mode='binary',
																  shuffle = False)

	test_generator = test_datagen.flow_from_directory(path_to_validation,target_size=(img_size, img_size),
																  classes = classes_source,
																  batch_size=batch_size_source,
																  class_mode='binary',
																  shuffle = False)

	return train_generator,validation_generator,test_generator

def save_bottleneck_features(model,train_generator,validation_generator,test_generator,trainNum,valNum,testNum,batch_size_source,recompute_transfer_values,verbose,**kwargs):
	"""
	Saves the bottleneck features of train set, validation set and test set in a file on hard disc. It uses the data from generators to train and get the result from training the model which are the transfer values. 
	
	#Inputs:
	Model: model to train the data 
	Train_generator : the train set after the generation of the data 
	validation_generator: the validation set after the generation of the data
	test_generator: the test set after the generation of the data
	trainNum: the number of examples to train in the train set
	valNum: the number of examples to train in the validation set
	testNum: the number of examples to train in the test set
	batch_size_source: the size of batch from the domain of source
	recompute_transfer_values: a Boolean whether the transfer values have been recomputed or not. It is False as initialization. 
	Verbose: a Boolean to control whether to print some details or not. 
	
	#Output:
	A npy file with all the bottleneck values.
	"""
	file1 = Path('bottleneck_features_train.npy')
	if not file1.is_file() or recompute_transfer_values:
		if verbose:
			print('bottleneck_features_train.npy')
		bottleneck_features_train = model.predict_generator(train_generator, trainNum // batch_size_source, use_multiprocessing=False, verbose=1)
		np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


	file2 = Path('bottleneck_features_val.npy')
	if not file2.is_file() or recompute_transfer_values:
		if verbose:
			print('bottleneck_features_val.npy')
		bottleneck_features_val = model.predict_generator(validation_generator, valNum // batch_size_source, use_multiprocessing=False, verbose=1)
		np.save(open('bottleneck_features_val.npy', 'wb'), bottleneck_features_val)

	file3 = Path('bottleneck_features_test.npy')
	if not file3.is_file() or recompute_transfer_values:
		if verbose:
			print('bottleneck_features_test.npy')
		bottleneck_features_test = model.predict_generator(test_generator, testNum // batch_size_source, use_multiprocessing=False, verbose=1)
		np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)

def top_layer_builder(num_of_classes,lr_source,**kwargs):
	"""
	Builds the top layer of the model of domain of source. The top layer classifies the inputs. It has 6 layers and it turns the 2048 neurons into 2 neurons which presents the labels of inputs. 
	#Input:
	num_of_classes: the number of classes of the results 
	lr_source: the learning rate of source
	#Output
	Model: the model of the top layers 
	"""
	train_data = np.load(open('bottleneck_features_train.npy',"rb"))
	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	#model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
	model.compile(optimizer = optimizers.Adam(lr=lr_source,amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
	return model

def top_layer_trainer(top_model,trainNum,valNum,testNum,train_generator,validation_generator,test_generator,batch_size_source,path_to_best_model,epochs_source,train_top_model,verbose,**kwargs):
	"""
	Trains the model of top layers. Changes the values inside the architecture of the model and saves the model
	
	#Input:
	top_model: the model of top layers
	trainNum: the number of examples to train in the train set
	valNum: the number of examples to train in the validation set
	testNum: the number of examples to train in the test set
	Train_generator : the train set after the generation of the data 
	validation_generator: the validation set after the generation of the data
	test_generator: the test set after the generation of the data
	batch_size_source: the size of batch of the domain of source
	path_to_best_model: the path to load the best model
	epochs_source: the epochs of the domain of source
	train_top_model: a Boolean to justify whether the model has been trained or not, which is False as initialization
	Verbose: a Boolean to control whether to print some details or not. 
	
	#Output:
	Print the values of the model

	"""
	file_exists = False
	file = Path(path_to_best_model)
	if file.is_file():
		file_exists = True

	if not file_exists or train_top_model :
		train_data = np.load(open('bottleneck_features_train.npy',"rb"))

		validation_data = np.load(open('bottleneck_features_val.npy',"rb"))

		test_data = np.load(open('bottleneck_features_val.npy',"rb"))

		train_labels,validation_labels,test_labels = train_generator.classes[:trainNum],validation_generator.classes[:valNum],test_generator.classes[:testNum]

		earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')

		checkpoint = ModelCheckpoint(path_to_best_model, monitor='val_acc', verbose=1, save_best_only=True, period=1,mode='max')

		top_model.fit(train_data, train_labels,
				  epochs=epochs_source,
				  batch_size=batch_size_source,
				  validation_data=(validation_data, validation_labels),
				  callbacks = [earlystop,checkpoint],
				  shuffle = True)

		print(top_model.evaluate(test_data, test_labels, verbose=verbose))

def full_model_builder(bottom_model,top_model,lr_source,path_to_best_model,**kwargs):
	"""
	Builds the fullmodel by combining the top layers and the bottom layers together. It loads the weights from the hard disc. 
	
	#Inputs:
	bottom_model: the bottom model has been constructed before
	top_model: the top model of the domain of source
	lr_source: the learning rate of domain of source
	path_to_best_model: path of reading the best model
	
	#Output:
	Full_model: the best full model from the domain of source

	"""
	top_model.load_weights(path_to_best_top_model)
	full_model = Model(inputs= bottom_model.input, outputs= top_model(bottom_model.output))
	sgd = optimizers.SGD(lr=lr_source, decay=1e-6, momentum=0.9, nesterov=True)
	for layer in full_model.layers:
		layer.trainable = False
	return full_model
	full_model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])

def fine_tune_builder(based_model_last_block_layer_number,lr_source,**kwargs):
	"""
	Builds a fine tuning model. It uses the values by using the previous model of domain of source. 
	#Inputs: 
	based_model_last_block_layer_number: the number of layers from the bottom model which will be blocked 
	lr_source: the learning rate of domain of source
	#Output:
	Model: the model of fine-tune 
	"""
	
	k.clear_session()
	model = customModelLoader("full_model_architecture.json","full_model_weights.h5")
	for layer in model.layers[:based_model_last_block_layer_number]:
		layer.trainable = False
	for layer in model.layers[based_model_last_block_layer_number:]:
		layer.trainable = True
	adam = optimizers.Adam(lr=lr_source, amsgrad=True)
	model.compile(optimizer = adam, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def fine_tune_trainer(model,train_generator_source,validation_generator_source,test_generator_source,path_to_best_model,lr_source,epochs_source,batch_size_source,**kwargs):
	
	"""
	Trains the fine tune model. 
	
	#Input:
	Model: model to train the data 
	train_generator_source: the train set after the generation of the data from domain of source 
	validation_generator_source: the validation set after the generation of the data from domain of source 
	test_generator_source: the test set after the generation of the data from the domain of source
	path_to_best_model: the path of the best model
	lr_source: the learning rate of domain of source
	epochs_source: epoch of the domain of source. 
	
	#Output
	the trained model which has the same architecture of precedent model but with different weights. 
	"""
	
	earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=10, verbose=1, mode='auto')
	checkpoint = ModelCheckpoint(path_to_best_model, monitor='val_acc', verbose=1, save_best_only=True, period=1,mode='max')
	model.fit_generator(train_generator_source,validation_data=validation_generator_source,verbose=1,callbacks=[earlystop,checkpoint],epochs=epochs_source)
	score = model.evaluate_generator(test_generator_source)
	print(model.metrics_names,score)

############################################################################
# Initializer FIRST LAYERS                                                #
############################################################################

def first_layers_reinitializer(model,layerLimit,**kwargs):
	"""
	re-initializes weights of layers up to layerLimit
	#Input:
	model:  the full model
	layerLimit: the number of layer which will be blocked
	#Output:
	model: the model which has been initialized
	"""

	for layer in model.layers[:layerLimit]:
		layer.trainable = True
#		session = k.get_session()
#		for v in layer.__dict__:
#			v_arg = getattr(layer,v)
#			if hasattr(v_arg,'initializer'):
#				initializer_method = getattr(v_arg,'initializer')
#				initializer_method.run(session=session)
				#print('reinitializing layer {}.{}'.format(layer.name, v))
	for layer in model.layers[layerLimit:]:
		layer.trainable = False
	sgd = optimizers.SGD(lr=lr_target, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])
	return model

def small_net_builder(originalSize,resizeFactor,lr_target,**kwargs):
	"""
	Build a simple CNN of 3 convolutional layers and 3 fully connected layers with dropout 
	#Input:
	originalSize: the original size of images
	resizeFactor: the resize factor 
	lr_target: learning rate of domain of target 
	#Output:
	model : the small CNN model
	"""
	img_size = originalSize*resizeFactor

	if k.image_data_format() == 'channels_first':
		input_shape = (3, img_size, img_size)
	else:
		input_shape = (img_size, img_size, 3)

	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(34))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	sgd = optimizers.SGD(lr=lr_target, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(optimizer = sgd, loss='binary_crossentropy', metrics=['accuracy'])

	return model

def from_generator_to_array(path_to_train,path_to_validation,trainNum,valNum,testNum,classes_target,originalSize,resizeFactor,transformation_ratio,**kwargs):
	"""
	Generate image data batches in array format, return training, validation and test datasets
	#Input:
	path_to_train: the path to the train set 
	path_to_validation: the path to the validation set 
	trainNum: the number of train examples
	valNum: the number of validation examples
	testNum: the number of test examples
	classes_target: the classes of target domain
	originalSize: the original size of image
	resizeFactor: the resize factor
	transformation_ratio: the ratio of transformation for rotation, zoom, etc. 
	#Output:
	x_train,y_train,x_val,y_val,x_test,y_test: the X, Y arrays of train, validation, test set. 
	"""
	img_size = originalSize*resizeFactor

	train_datagen = ImageDataGenerator(rescale=1. / 255)

	validation_datagen = ImageDataGenerator(rescale=1. / 255)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_directory(path_to_train,target_size=(img_size, img_size),
														batch_size=trainNum,
														classes = classes_target,
														class_mode='binary',
														shuffle = False)

	validation_generator = validation_datagen.flow_from_directory(path_to_validation,target_size=(img_size, img_size),
																  classes = classes_target,
																  batch_size=valNum,
																  class_mode='binary',
																  shuffle = False)

	test_generator = test_datagen.flow_from_directory(path_to_validation,target_size=(img_size, img_size),
																  classes = classes_target,
																  batch_size=testNum,
																  class_mode='binary',
																  shuffle = False)
	x_train,y_train = train_generator.next()
	x_val,y_val = validation_generator.next()
	x_test,y_test = test_generator.next()

	return x_train,y_train,x_val,y_val,x_test,y_test

#######################################################
#               SAVING PART                           #
#######################################################

def saveModelStructure(modelArchitecturePath,model):
	"""
	Save a model structure and its path
	#Input:
	modelArchitecturePath: the path of model's architecture
	model: the new model
	#Output:
	save the model into a file
	"""
	
	architecture = model.to_json()
	with open(modelArchitecturePath, 'wb') as f:
		pickle.dump(architecture, f)

def saveModelWeigths(modelWeigthPath,model):
	"""
	Save a model’s weights and its path
	#Input:
	modelWeigthPath: the path of model's weights
	model
	#Output
	save the path 
	"""
	
	model.save_weights(modelWeigthPath)

def customModelLoader(modelArchitecturePath,modelWeigthPath):
	"""
	Load a model with its architecture and its weights
	#Input:
	modelArchitecturePath: the path of model's architecture
	modelWeigthPath: the path of model's weights
	#Output 
	model: loaded model
	"""
	with open(modelArchitecturePath, 'rb') as f1:
		architecture = pickle.load(f1)
	model  = model_from_json(architecture)
	model.load_weights(modelWeigthPath)
	#sgd = optimizers.SGD(lr=lr_target, decay=1e-6, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=lr_target, amsgrad=True)
	model.compile(optimizer = adam, loss='binary_crossentropy', metrics=['accuracy'])
	return model


#######################################################
#               BOOSTING                             #
#######################################################
def take(tab,indexes):
	"""
	Reorder the table based on indexes, return the new table
	#Input:
	tab: table
	indexes:  the indexes of tables
	#output
	output: the new table
	"""
	output = np.zeros(tab.shape)
	c=0
	for i in indexes:
		output[c] = tab[i]
		c+=1
	return output

def batchBooster(x_train,y_train,x_val,y_val,x_test,y_test,params_temp,epochs_target,lr_target,batch_size_target,threshold,layerLimit,times,bigNet,originalSize,resizeFactor,proba_threshold,step,verbose,**kwargs):
	"""
	Conduct the transboost process and obtain a list of weak trained models.

	#Input
	x_train: Array of training data
	y_train: Array of training data label values
	x_val: Array of validation data
	y_val: Array of validation data label values
	x_test: Array of test data
	y_test: Array of test data label values
	params_temp: Dictoinary with configs of the experiment
	epochs_target: Maximal epoches for a training process on the same dataset
	lr_target: Learning rate for the training on the target dataset
	batch_size_target: Size of batch for training data
	threshold: Threshold for distinguish if a weak model has been obtained
			   The training process won't stop until the validation accuracy is above the threshold.
	layerLimit: Integer value for marking which layers are trainable.
	times: Repetition times, each time a weak model is obtained.
	bigNet: Boolean value to decide on which net the experiment works
	originalSize: Original size of data image
	resizeFactor: Factor used to resize the image size
	proba_threshold: Threshold for distinguish which class the data belongs to. 
					 If the prediction proba is above the threshold, the data is predicted as Class 1, else as Class 2.
	step: Integer value, for every "step" repititions, we see the progress of the model's performance
	verbose: Boolean value to control the display of printing words in the console.

	#Output
	model_list: List containing all trained models' path which are stocked in the disk.
	error_list: List containing the training errors related to each trained model.
	alpha_list: List of weights related to each model.
    It is used when making a weighted linear combination of each weak model as the final strong model.

	"""
	train_length = len(x_train)
	model_list = []
	error_list = []
	alpha_list = []

	if train_length==0:
		raise NameError("length of training set equals 0")

	prob = np.repeat(1/train_length, train_length)
	indexes = list(range(train_length))

	k.clear_session()

	for time in range(times):
#       print("="*50)
#       print( "boosting step number "+str(time))
		current_model_path = os.path.join(models_weights_path,"model_"+str(time)+".h5")
		train_boost_indexes = np.random.choice(indexes,p=prob,size=train_length,replace=True)

		x_train_boost = take(x_train,train_boost_indexes)
		y_train_boost = take(y_train,train_boost_indexes)

		if bigNet :
			current_model = customModelLoader("full_model_architecture.json","full_model_weights.h5")
			current_model = first_layers_reinitializer(current_model,layerLimit)
		else :
			current_model = small_net_builder(originalSize,resizeFactor,lr_target)

		error = 0
		while error == 1 or error == 0 :
			if bigNet :
				current_model = customModelLoader("full_model_architecture.json","full_model_weights.h5")
				current_model = first_layers_reinitializer(current_model,layerLimit)
			else:
				current_model = small_net_builder(originalSize,resizeFactor,lr_target)

			current_model.fit(x_train_boost, y_train_boost, batch_size = batch_size_target, epochs=epochs_target, validation_split = .1, verbose=verbose, callbacks=[callbackBoosting(threshold,"acc",current_model_path,verbose)], shuffle=False)

			error = 1 - current_model.evaluate(x_train, y_train, verbose=0)[1]

		#saveModelWeigths(current_model_path,current_model)

		alpha = .5*np.log((1-error)/error)

		error_list.append(error)

		model_list.append(current_model_path) #adds model path to list

		alpha_list.append(alpha)

		predicted_probs = current_model.predict(x_train)
		predicted_classes = []

		for predicted_prob in predicted_probs:
			if predicted_prob >= proba_threshold:
				predicted_classes.append(1)
			else :
				predicted_classes.append(0)

		for i in range(len(predicted_classes)):
			if predicted_classes[i] == y_train[i]:
				prob[i] = 1/(2*(1-error))
			else:
				prob[i] = 1/(2*error)

		prob = prob / np.sum(prob)

		if (time+1) % step == 0:
			predicted_classes = prediction_boosting(x_test,model_list,alpha_list,**params_temp)
			print("time: ",time+1,"accuracy :",accuracy(y_test,predicted_classes))

		del current_model
		gc.collect() #garbage collector frees up memory (normally)
		k.clear_session()

	return model_list, error_list, alpha_list

def prediction_boosting(x,model_list, alpha_list,proba_threshold,**kwargs):
	"""
	Give prediction of classes which the test data belong to from the trained models.
	Used with accuracy() in batchBooster() to see the progress of the model's performance while training and in main function to see the final model's performance.

	#Input
	x: Array of test data.
	model_list: List of trained and stocked models' paths.
	alpha_list: List of weights related to each model.
				It is used when making a weighted linear combination of each weak model as the final strong model.
	proba_threshold: Threshold for distinguish which class the data belongs to. 
					 If the prediction proba is above the threshold, the data is predicted as Class 1, else as Class 2.

	#Output
	results: Array of predicted label values matching the test data
		"""
	k.clear_session()
	n_samples = len(x)
	n_models = len(model_list)
	results = []
	predicted_class_list = []
	modelArchitecturePath = "full_model_architecture.json"
	for model_name in model_list:
		model = customModelLoader(modelArchitecturePath,model_name)
		probas = np.array(model.predict(x))
		booleans = probas >= proba_threshold
		booleans = list(chain(*booleans))
		to_append = []
		for boolean in booleans:
			if boolean:
				to_append.append(1)
			else:
				to_append.append(-1)
		predicted_class_list.append(to_append)
		del model
		gc.collect()
		k.clear_session()

	predicted_class_list = np.array(predicted_class_list)
	predicted_class_list.reshape((n_models,n_samples))
	predicted_class_list = np.transpose(predicted_class_list)
	alpha_list = np.array(alpha_list)
	raw_results = np.dot(predicted_class_list,alpha_list)

	for raw_result in raw_results:
		if raw_result >=0:
			results.append(1)
		else:
			results.append(0)
	return results

def accuracy(y_true,y_pred,**kwargs):
	"""
	Mesure the accuracy of prediction from the trained model.
	Used in batchBooster() to see the progress of the model's performance while training and in main function to see the final model's performance.

	#Input
	y_true: Array of real label values matching the test data.
	y_pred: Array of predicted label values matching the test data.

	#Output
	accuracy: Accuracy of prediction.
	"""
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
	""" 
	this function stands for testing purposes
	"""

	dt = datetime.now()
	print(200*"#")
	print("TRANSBOOST LAUNCH on {:%Y-%m-%d %H:%M}".format(datetime.now()))
	print(200*"#")

	downloader(url,path) #path ad url in dataLoader.py
	params = getArgs()

	print("\n\n\nexecuted with the following settings:\n")
	printParams(params)
	print("\n\n\n")

	num_of_classes = len(params['classes_source'])
	based_model_last_block_layer_number = 132

	checkDir(params['models_path'])
	checkDir(params['models_weights_path'])

	try:
		bottom_model = bottom_layers_builder(**params)
		train_generator_source,validation_generator_source,test_generator_source = create_generators(path_to_train,path_to_validation,**params)
		save_bottleneck_features(bottom_model,train_generator_source,validation_generator_source,test_generator_source,trainNum_source,valNum_source,testNum_source,**params)
		# top_model = top_layer_builder(num_of_classes,**params)
		# top_layer_trainer(top_model,trainNum_source,valNum_source,testNum_source,train_generator_source,validation_generator_source,test_generator_source,**params)
		# top_model_init = top_layer_builder(num_of_classes,**params)
		# full_model = full_model_builder(bottom_model,top_model_init,**params)
		
		# saveModelStructure("full_model_architecture.json",full_model)
		# saveModelWeigths("full_model_weights.h5",full_model)
		# del full_model
		# gc.collect()
		# k.clear_session()

		model = fine_tune_builder(based_model_last_block_layer_number,**params)
		fine_tune_trainer(model,train_generator_source,validation_generator_source,test_generator_source,**params)
		del model
		gc.collect()
		k.clear_session()

		#2nd part
		# x_train_target,y_train_target,x_val_target,y_val_target,x_test_target,y_test_target = from_generator_to_array(path_to_train,path_to_validation,trainNum_target,valNum_target,testNum_target,**params)
		# model_list, _ , alpha_list = batchBooster(x_train_target,y_train_target,x_val_target,y_val_target,x_test_target,y_test_target,params,**params)
		# model_list=['models_weights/model_0.h5']
		# alpha_list=[1]
		# predicted_classes = prediction_boosting(x_train_target,model_list,alpha_list,**params)
		# print("Final accuracy train:",accuracy(y_train_target,predicted_classes))		
		# predicted_classes = prediction_boosting(x_test_target,model_list,alpha_list,**params)
		# print("Final accuracy :",accuracy(y_test_target,predicted_classes))

	except MemoryError:
		objects = [o for o in gc.get_objects()]
		for o in objects:
			print(o, sys.getsizeof(o))

if __name__ == '__main__':
	main()
