"""
# @Author: Luc Blassel <zlanderous>
# @Date:   2018-01-15T00:21:20+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-15T14:31:14+01:00

Romain Gautron
"""
from dataProcessing import *
from binariser import *
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense
from keras import backend as k
import paramGetter as pg
import sys

if len(sys.argv==0):
    print("please specify path of config file...")
    sys.exit()

path = sys.argv[0]
params = pg.reader(path) #dictionary with relevant parameters

img_width, img_height = 139, 139
epochs = 50

from callbackBoosting import *

#####################
# LOADING DATA		#
#####################

def loader(wantedLabels,trainnum,testnum):
	""" this function loads the datasets from CIFAR 10 with correct output in order to feed inception
	OUTPUTQ : 32*resizefactor,32*resizefactor"""
	raw_train,raw_test = loadRawData()
	x_train,y_train = loadTrainingData(raw_train,wantedLabels,trainnum)
	x_test,y_test = loadTestingData(raw_test,wantedLabels,testnum)
	y_train_bin,y_test_bin = binarise(y_train),binarise(y_test)
	return x_train, y_train_bin, x_test, y_test_bin

#####################################
# BUILDING MODEL FOR TWO CLASSES	#
#####################################

def full_model_builder(img_width,img_height):
	""" this function builds a model that outputs binary classes
	INPUTS : 
	- img_width >=139
	- img_height >=139
	OUTPUTS :
	-full model
	"""
	model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

	# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
	for layer in model.layers:
	    layer.trainable = False

	#Adding custom Layers
	x = model.output
	x = Flatten()(x)
	x = Dense(1024, activation="relu")(x)
	predictions = Dense(2, activation="softmax")(x)

	# creating the final model
	model_final = Model(input = model.input, output = predictions)

	# compile the model
	model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

	return model_final

#####################################
# TRAINING AND TESTING FULL MODEL	#
#####################################

def full_model_trainer(model,x_train,y_train_bin,x_test,y_test_bin,epochs):
	"""
	this function's purpose is to train the full model
	INPUTS : the model to train
	OUPUTS : the model score
	"""
	model.fit(x = x_train, y = y_train_bin, batch_size = 10, epochs = epochs,validation_split = 0.1)
	score = model.evaluate(x_test, y_test_bin, verbose=1)
	return score


############################################################################
# TRAINING FIRST LAYERS 												   #
############################################################################

def first_layers_modified_model_builder(model,layer_limit):
	"""this function changes a model whose first layers are trainable with reinitialized weights
	INPUTS : 
	- model to modifiy
	- layer_limit : limit of the first layer to modify (see layer.name)
	OUTPUTS :
	- copy of the modified model
	"""
	model_copy =  model
	for layer in model_copy.layers[:layer_limit]:
	    layer.trainable = True
	    previous_weights = layer.get_weights()
	    new_weights = list((10*np.random.random((np.array(previous_weights).shape))))
	    layer.set_weights(new_weights)

	for layer in model_copy.layers[layer_limit:]:
	    layer.trainable = False
	return model_copy

def first_layers_modified_model_trainer(model,x_train,y_train_bin,epochs,threshold):
	"""
	this function trains models from [first_layers_modified_model_builder] function
	"""
	model.fit(x = x_train, y = y_train_bin, batch_size = 10, epochs = epochs,validation_split = 0.1,callbacks = [callbackBoosting(threshold)])


############################################################################
# MAIN 																	   #
############################################################################

def main():
	""" this function stands for testing purposes
	"""
	trainnum = 1000
	testnum = 1000
	wantedLabels = ['dog','truck']
	img_width, img_height = resizeFactor*32,resizeFactor*32
	epochs = 1
	threshold = .2
	layer_limit =  10

	x_train, y_train_bin, x_test, y_test_bin = loader(wantedLabels,trainnum,testnum)
	print("data loaded")
	full_model = full_model_builder(img_width,img_height)
	print("full model built")
	score = full_model_trainer(full_model,x_train,y_train_bin,x_test,y_test_bin,epochs)
	print("modified model trained")
	print("full model score ",score)
	modified_model = first_layers_modified_model_builder(full_model,layer_limit)
	print("modified model built")
	first_layers_modified_model_trainer(modified_model,x_train,y_train_bin,epochs,threshold)
	print("modified model trained")

if __name__ == '__main__':
	main()

