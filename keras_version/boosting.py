# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-01-15 14:59:20
# @Last modified by:   Luc Blassel
# @Last modified time: 2018-01-28T15:57:45+01:00
"""
inspired by https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
"""
import numpy as np
import time
import os
from binariser import *
from dataProcessing import *
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Activation
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator
from  callbackBoosting import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from dataLoader import *
from pathlib import Path
from keras.utils.np_utils import to_categorical
import pandas as pd
import copy as cp
import _pickle as pickle
from itertools import chain

downloader(url,path)
models_path = "models"

# checks if models directory already exists, and iuf not creates it

dataPath = models_path
if not os.path.exists(dataPath):
	print("creating" ,dataPath, "directory")
	os.makedirs(dataPath)


#####################################
# BUILDING MODEL FOR TWO CLASSES    #
#####################################

def bottom_layers_builder(originalSize,resizeFactor):
	"""
	romain.gautron@agroparistech.fr
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

def create_generators(classes,path_to_train,path_to_validation,originalSize,resizeFactor,batch_size,transformation_ratio):
	"""
	romain.gautron@agroparistech.fr
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
														batch_size=batch_size,
														classes = classes,
														class_mode='binary',
														shuffle = False)

	validation_generator = validation_datagen.flow_from_directory(path_to_validation,target_size=(img_size, img_size),
																  classes = classes,
																  batch_size=batch_size,
																  class_mode='binary',
																  shuffle = False)

	test_generator = test_datagen.flow_from_directory(path_to_validation,target_size=(img_size, img_size),
																  classes = classes,
																  batch_size=batch_size,
																  class_mode='binary',
																  shuffle = False)

	return train_generator,validation_generator,test_generator

def save_bottleneck_features(model,train_generator,validation_generator,test_generator,trainNum,valNum,testNum,batch_size,recompute_transfer_values):
	"""
	romain.gautron@agroparistech.fr
	"""
	file1 = Path('bottleneck_features_train.npy')
	if not file1.is_file() or recompute_transfer_values:
		print('bottleneck_features_train.npy')
		bottleneck_features_train = model.predict_generator(train_generator, trainNum // batch_size, use_multiprocessing=False, verbose=1)
		np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


	file2 = Path('bottleneck_features_val.npy')
	if not file2.is_file() or recompute_transfer_values:
		print('bottleneck_features_val.npy')
		bottleneck_features_val = model.predict_generator(validation_generator, valNum // batch_size, use_multiprocessing=False, verbose=1)
		np.save(open('bottleneck_features_val.npy', 'wb'), bottleneck_features_val)

	file3 = Path('bottleneck_features_test.npy')
	if not file3.is_file() or recompute_transfer_values:
		print('bottleneck_features_test.npy')
		bottleneck_features_test = model.predict_generator(test_generator, testNum // batch_size, use_multiprocessing=False, verbose=1)
		np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test)

def top_layer_builder(lr,num_of_classes):
	"""
	romain.gautron@agroparistech.fr
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
	model.compile(optimizer = optimizers.Adam(lr=lr,amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
	return model

def top_layer_trainer(train_top_model,top_model,epochs,batch_size,trainNum,valNum,testNum,lr,train_generator,validation_generator,test_generator,path_to_best_model):
	"""
	romain.gautron@agroparistech.fr
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

		checkpoint = ModelCheckpoint(path_to_best_model, monitor='val_loss', verbose=1, save_best_only=True, period=1,mode='max')

		top_model.fit(train_data, train_labels,
				  epochs=epochs,
				  batch_size=batch_size,
				  validation_data=(validation_data, validation_labels),
				  callbacks = [earlystop,checkpoint],
				  shuffle = True)

		print(top_model.evaluate(test_data, test_labels, verbose=1))

def full_model_builder(path_to_best_top_model,bottom_model,top_model,lr):
	"""
	romain.gautron@agroparistech.fr
	"""
	top_model.load_weights(path_to_best_top_model)
	full_model = Model(inputs= bottom_model.input, outputs= top_model(bottom_model.output))
	full_model.compile(optimizer = optimizers.Adam(lr=lr,amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
	for layer in full_model.layers:
		layer.trainable = False
	return full_model



############################################################################
# TRAINING FIRST LAYERS                                                    #
############################################################################

def first_layers_modified_model_builder(model,layerLimit,reinitialize_bottom_layers ):
	"""
	romain.gautron@agroparistech.fr

	this function changes a model whose first layers are trainable with reinitialized weights
	INPUTS :
	- model to modifiy
	- layerLimit : limit of the first layer to modify (see layer.name)
	OUTPUTS :
	- copy of the modified model
	"""
	model_copy =  cp.deepcopy(model)
	for layer in model_copy.layers[:layerLimit]:

		session = k.get_session()
		layer.trainable = True
		if reinitialize_bottom_layers :
			for v in layer.__dict__:
				v_arg = getattr(layer,v)
				if hasattr(v_arg,'initializer'):
					initializer_method = getattr(v_arg, 'initializer')
					initializer_method.run(session=session)
					print('reinitializing layer {}.{}'.format(layer.name, v))

	for layer in model_copy.layers[layerLimit:]:
		layer.trainable = False

	return model_copy

def first_layers_modified_model_trainer(model,train_generator,validation_generator,test_generator,epochs,threshold):
	"""
	romain.gautron@agroparistech.fr
	this function trains models from [first_layers_modified_model_builder] function
	"""
	model.fit_generator(train_generator, epochs=epochs, verbose=1, callbacks=[callbackBoosting(threshold,"val_acc")], validation_data=validation_generator, use_multiprocessing=False, shuffle=True)
	score = model.evaluate_generator(test_generator)
	print("projector score : ", score)

def small_net_builder(originalSize,resizeFactor,lr):
	"""
	romain.gautron@agroparistech.fr
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

	model.compile(optimizer = optimizers.Adam(lr=lr,amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])

	return model

def from_generator_to_array(classes,path_to_train,path_to_validation,originalSize,resizeFactor,transformation_ratio,trainNum,valNum,testNum):
	"""
	romain.gautron@agroparistech.fr
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
														batch_size=trainNum,
														classes = classes,
														class_mode='binary',
														shuffle = False)

	validation_generator = validation_datagen.flow_from_directory(path_to_validation,target_size=(img_size, img_size),
																  classes = classes,
																  batch_size=valNum,
																  class_mode='binary',
																  shuffle = False)

	test_generator = test_datagen.flow_from_directory(path_to_validation,target_size=(img_size, img_size),
																  classes = classes,
																  batch_size=testNum,
																  class_mode='binary',
																  shuffle = False)
	x_train,y_train = train_generator.next()
	x_val,y_val = validation_generator.next()
	x_test,y_test = test_generator.next()

	return x_train,y_train,x_val,y_val,x_test,y_test


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
def booster(full_model,x_train,y_train,x_val,y_val,epochs,threshold,layerLimit,times,bigNet,originalSize,resizeFactor,lr,proba_threshold):
	"""
	romain.gautron@agroparistech.fr
	"""
	train_length = len(x_train)
	model_list = []
	error_list = []
	alpha_list = []
	c = 1

	if train_length==0:
		raise NameError("length of training set equals 0")

	prob = np.repeat(1/train_length, train_length)
	indexes = list(range(train_length))

	for time in range(times):

		current_model_path = os.join(models_path,"model_"+str(c)+"h5")

		train_boost_indexes = np.random.choice(indexes,p=prob,size=train_length,replace=True)
		x_train_boost = take(x_train,train_boost_indexes)
		y_train_boost = take(y_train,train_boost_indexes)

		if bigNet :
			current_model = first_layers_modified_model_builder(full_model,layerLimit)
		else :
			current_model = small_net_builder(originalSize,resizeFactor,lr)

		error = 0
		while error == 1 or error == 0 :
			if bigNet :
				current_model = first_layers_modified_model_builder(full_model,layerLimit)
			else:
				current_model = small_net_builder(originalSize,resizeFactor,lr)

			current_model.fit(x_train_boost, y_train_boost, epochs=epochs, verbose=1, callbacks=[callbackBoosting(threshold,"acc")], shuffle=True)

			error = 1 - current_model.evaluate(x_val, y_val, verbose=1)[1]
		alpha = .5*np.log((1-error)/error)

		error_list.append(error)
		# model_list.append(current_model)

		model_list.append(current_model_path) #adds model path to list
		current_model.save(current_model_path) #saves model to disk

		del current_model #frees up memory space

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
				prob[i] = prob[i]*np.exp(-alpha)
			else:
				prob[i] = prob[i]*np.exp(alpha)
		prob = prob / np.sum(prob)

	return model_list, error_list, alpha_list

def prediction_boosting(x,model_list, alpha_list,proba_threshold):
	"""
	romain.gautron@agroparistech.fr
	"""
	n_samples = len(x)
	n_models = len(model_list)
	results = []
	predicted_class_list = []
	c = 0
	for model_name in model_list:
		print("beginning prediction for model :",c)

		model = load_model(model_name) #loads model from disk
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
		print("ending prediction for model :",c)
		c +=1

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

def accuracy(y_true,y_pred):
	"""
	romain.gautron@agroparistech.fr
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
	""" this function stands for testing purposes
	"""
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

	trainNum_source = 7950
	valNum_source = 2040
	testNum_source = 2040
	trainNum_target = 8010
	valNum_target = 1980
	testNum_target = 1980

	lr_source = 0.0001
	epochs_source = 50

	recompute_transfer_values = False
	train_top_model = False

	bottom_model = bottom_layers_builder(originalSize,resizeFactor)
	train_generator_source,validation_generator_source,test_generator_source = create_generators(classes_source,path_to_train,path_to_validation,originalSize,resizeFactor,batch_size_source,transformation_ratio)
	pstest = pd.Series(test_generator_source.classes[:testNum_source])
	counts = pstest.value_counts()
	print("test classes ",counts)
	pstrain = pd.Series(train_generator_source.classes[:trainNum_source])
	counts = pstrain.value_counts()
	print("train classes ",counts)
	save_bottleneck_features(bottom_model,train_generator_source,validation_generator_source,test_generator_source,trainNum_source,valNum_source,testNum_source,batch_size_source,recompute_transfer_values)
	top_model = top_layer_builder(lr_source,num_of_classes)
	top_layer_trainer(train_top_model,top_model,epochs_source,batch_size_source,trainNum_source,valNum_source,testNum_source,lr_source,train_generator_source,validation_generator_source,test_generator_source,path_to_best_top_model)
	top_model_init = top_layer_builder(lr_source,num_of_classes)
	full_model = full_model_builder(path_to_best_top_model,bottom_model,top_model_init,lr_source)
	# full_model_score = full_model.evaluate_generator(test_generator_source)
	# print(full_model_score)

	layerLimit = 15
	epochs_target = 100
	lr_target = 0.0001
	batch_size_target = 10
	threshold = .65
	reinitialize_bottom_layers = False
	bigNet = False
	times = 100

	# train_generator_target,validation_generator_target,test_generator_target = create_generators(classes_target,path_to_train,path_to_validation,originalSize,resizeFactor,batch_size_target,transformation_ratio)
	# first_layers_modified_model = first_layers_modified_model_builder(full_model,layerLimit,reinitialize_bottom_layers)
	# first_layers_modified_model_score = first_layers_modified_model.evaluate_generator(test_generator_target)
	# print(first_layers_modified_model_score)
	# first_layers_modified_model_trainer(first_layers_modified_model,train_generator_target,validation_generator_target,test_generator_target,epochs_target,threshold)

	# small_net_builder= small_net_builder(originalSize,resizeFactor,lr)
	# first_layers_modified_model_trainer(small_net_builder,train_generator_target,validation_generator_target,test_generator_target,epochs_target,threshold)

	proba_threshold = .5
	x_train_target,y_train_target,x_val_target,y_val_target,x_test_target,y_test_target = from_generator_to_array(classes_target,path_to_train,path_to_validation,originalSize,resizeFactor,transformation_ratio,trainNum_target,valNum_target,testNum_target)
	model_list, error_list, alpha_list = booster(full_model,x_train_target,y_train_target,x_val_target,y_val_target,epochs_target,threshold,layerLimit,times,bigNet,originalSize,resizeFactor,lr_target,proba_threshold)
	# pickler = pickle.Pickler(open('alpha_list.pkl', 'wb'), -1)
	# pickler.dump(alpha_list)
	# print(model_list, error_list, alpha_list)
	# c = 0
	# for model in model_list:
	# 	model_path = "model"+ str(c) +".h5"
	# 	model.save(model_path)
	# 	c+=1
	predicted_classes = prediction_boosting(x_test_target,model_list, alpha_list,proba_threshold)
	# np.save(open('boosting_classes.npy', 'wb'), predicted_classes)
	print(accuracy(y_test_target,predicted_classes))

	# model_list = []
	# for time in range(times):
	# 	path_model = "model"+ str(time) +".h5"
	# 	model = load_model(path_model)
	# 	model_list.append(model)
	# with open('result_list.pkl', 'rb') as pickle_file:
	# 	alpha_list = pickle.load(pickle_file)

	# predicted_classes = prediction_boosting(x_test_target,model_list, alpha_list,proba_threshold)
	print(accuracy(y_test_target,predicted_classes))

if __name__ == '__main__':
	main()
