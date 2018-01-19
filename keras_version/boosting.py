# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-01-15 14:59:20
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-17T14:13:05+01:00

import numpy as np
import time
from binariser import *
from dataProcessing import *
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Activation
from keras import backend as k
from  callbackBoosting import *
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from dataLoader import *
from pathlib import Path
from keras.utils.np_utils import to_categorical

#####################
# LOADING DATA        #
#####################

# def loader(trainLabels,testLabels,trainNum,testNum,**kwargs):
#     """
#     this function loads the datasets from CIFAR 10 with correct output in order to feed inception
#     OUTPUT : 32*resizefactor,32*resizefactor
#     """

#     raw_train,raw_test = loadRawData()
#     x_train,y_train = loadTrainingData(raw_train,labels=trainLabels,trainCases=trainNum)
#     # x_train,y_train = loadTrainingData(raw_train,trLabels,trNum)
#     x_test,y_test = loadTestingData(raw_test,labels=testLabels,testCases=testNum)
#     # x_test,y_test = loadTestingData(raw_test,teLabels,teNum)
#     y_train_bin,y_test_bin = binarise(y_train),binarise(y_test)

#     return x_train, y_train_bin, x_test, y_test_bin

downloader(url,path)

#####################################
# BUILDING MODEL FOR TWO CLASSES    #
#####################################

def bottom_layers_builder(originalSize,resizeFactor,**kwargs):
    img_size = originalSize*resizeFactor
    #model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_size, img_size, 3))
    model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_size, img_size, 3))

    for layer in model.layers :
        layer.trainable = False
    return model

def create_generators(classes,path_to_train,path_to_validation,originalSize,resizeFactor,batch_size,transformation_ratio):
    img_size = originalSize*resizeFactor

    # train_datagen = ImageDataGenerator(rescale=1. / 255,
    #                                    rotation_range=transformation_ratio,
    #                                    shear_range=transformation_ratio,
    #                                    zoom_range=transformation_ratio,
    #                                    cval=transformation_ratio,
    #                                    horizontal_flip=True,
    #                                    vertical_flip=True)

    train_datagen = ImageDataGenerator(rescale=1. / 255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

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

def save_bottleneck_features(model,train_generator,validation_generator,test_generator,trainNum,valNum,testNum,batch_size,recompute):

    if not recompute :
        print('bottleneck_features_train.npy')
        file1 = Path('bottleneck_features_train.npy')
        if not file1.is_file():
            bottleneck_features_train = model.predict_generator(train_generator, trainNum // batch_size, use_multiprocessing=False, verbose=1)
            np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

        print('bottleneck_features_val.npy')
        file2 = Path('bottleneck_features_val.npy')
        if not file2.is_file():
            bottleneck_features_val = model.predict_generator(validation_generator, valNum // batch_size, use_multiprocessing=False, verbose=1)
            np.save(open('bottleneck_features_val.npy', 'wb'), bottleneck_features_val)

        print('bottleneck_features_test.npy')
        file3 = Path('bottleneck_features_test.npy')
        if not file3.is_file():
            bottleneck_features_test = model.predict_generator(test_generator, testNum // batch_size, use_multiprocessing=False, verbose=1)
            np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test) 

    else :

        bottleneck_features_train = model.predict_generator(train_generator, trainNum // batch_size, use_multiprocessing=False, verbose=1)
        np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)
        bottleneck_features_val = model.predict_generator(validation_generator, valNum // batch_size, use_multiprocessing=False, verbose=1)
        np.save(open('bottleneck_features_val.npy', 'wb'), bottleneck_features_val)
        bottleneck_features_test = model.predict_generator(test_generator, testNum // batch_size, use_multiprocessing=False, verbose=1)
        np.save(open('bottleneck_features_test.npy', 'wb'), bottleneck_features_test) 

def top_layer_builder(lr,num_of_classes):
    train_data = np.load(open('bottleneck_features_train.npy',"rb"))
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
    #model.compile(optimizer = optimizers.Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def top_layer_trainer(top_model,top_model_weights_path,epochs,batch_size,trainNum,valNum,testNum,lr):
    train_data = np.load(open('bottleneck_features_train.npy',"rb"))
    train_labels = np.array([0] * int(trainNum//2) + [1] * int(trainNum//2))
    #train_labels_binary = to_categorical(train_labels)

    validation_data = np.load(open('bottleneck_features_val.npy',"rb"))
    validation_labels = np.array([0] * int(valNum//2) + [1] * int(valNum//2))
    #validation_labels_binary = to_categorical(train_labels)

    test_data = np.load(open('bottleneck_features_val.npy',"rb"))
    test_labels = np.array([0] * int(testNum//2) + [1] * int(testNum//2))
    #test_labels_binary = to_categorical(test_labels)

    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')

    top_model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
              #callbacks = [earlystop])

    print(top_model.evaluate(test_data, test_labels, verbose=1))

    top_model.save_weights(top_model_weights_path)

def full_model_builder(bottom_model,top_model,lr):
    full_model = Model(inputs= bottom_model.input, outputs= top_model(bottom_model.output))
    full_model.compile(optimizer = optimizers.Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])
    return full_model

#####################################
# TRAINING AND TESTING FULL MODEL    #
#####################################

# def full_model_trainer(model,x_train,y_train_bin,x_test,y_test_bin,epochs,**kwargs):
#     """
#     this function's purpose is to train the full model
#     INPUTS : the model to train
#     OUPUTS : the model score
#     """
#     earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')

#     model.fit(x = x_train, y = y_train_bin, batch_size = 32, epochs = epochs, validation_split = .1, callbacks = [earlystop])
#     score = model.evaluate(x_test, y_test_bin, verbose=1)
#     return score


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
    model.fit(x = x_train, y = y_train_bin, batch_size = 64, epochs = epochs,callbacks = [callbackBoosting(threshold)])


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
    classes = ['dog','truck']
    num_of_classes = len(classes)
    top_model_weights_path = 'fc_model.h5'
    batch_size = 32
    transformation_ratio = .05
    originalSize = 32
    resizeFactor = 5
    path_to_train = path + "train"
    path_to_validation = path + "validation"
    path_to_test = path + "test"
    trainNum = 1024
    valNum = 512
    testNum = 512
    top_model_weights_path = 'bottleneck_fc_model.h5'
    lr = 0.0001
    epochs = 1000
    recompute = False
    bottom_model = bottom_layers_builder(originalSize,resizeFactor)
    train_generator,validation_generator,test_generator = create_generators(classes,path_to_train,path_to_validation,originalSize,resizeFactor,batch_size,transformation_ratio)
    save_bottleneck_features(bottom_model,train_generator,validation_generator,test_generator,trainNum,valNum,testNum,batch_size,recompute)
    top_model = top_layer_builder(lr,num_of_classes)
    top_layer_trainer(top_model,top_model_weights_path,epochs,batch_size,trainNum,valNum,testNum,lr)
    full_model = full_model_builder(bottom_model,top_model,lr)
    probas = full_model.predict_generator(test_generator, testNum // batch_size, use_multiprocessing=True, verbose=1)
    y_classes = probas.argmax(axis=-1)
    print(y_classes)

    # full_model = full_model_builder(img_width,img_height)
    # layerLimit = 10
    # epochs = 1
    # threshold = .5
    # times = 5
    # model_list, error_list, alpha_list = booster(full_model,x_train,y_train_bin,epochs,threshold,layerLimit,times)
    # print("model_list ", model_list)
    # print("error_list ", error_list)
    # print("alpha_list ", alpha_list)

if __name__ == '__main__':
	main()
