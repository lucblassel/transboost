# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-01-23 16:25:00
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-01-23 16:32:07
import numpy as np
import time
from binariser import *
from dataProcessing import *
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model,load_model
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

def prediction_boosting(x,model_list, alpha_list,proba_threshold):
	"""
	romain.gautron@agroparistech.fr
	"""
	n_samples = len(x)
	n_models = len(model_list)
	results = []
	predicted_class_list = []
	c = 0
	for model in model_list:
		print("beginning prediction for model :",c)
		probas = np.array(model.predict(x))
		print("probas : ", probas)
		to_append = []
		for proba in probas:
			if proba >= proba_threshold:
				predicted_class == 1
			else:
				predicted_class = -1
			to_append.append(predicted_class)
		predicted_class_list.append(to_append)
		print("to_append : ", to_append)
		print("ending prediction for model :",c)
		c +=1

	predicted_class_list = np.array(predicted_class_list)
	predicted_class_list.reshape((n_models,n_samples))
	predicted_class_list = np.transpose(predicted_class_list)
	alpha_list = np.array(alpha_list)
	raw_results = np.dot(predicted_class_list,alpha_list)

	for raw_result in raw_results:
		if raw_result >=0:
			results.append(0)
		else:
			results.append(-1)
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
	model_list = []
	C = 12
	for c in range(C):
		path = "model"+ str(c) +".h5"
		model = load_model(path)
		model_list.append(model)
	error_list,alpha_list = pickle.load('result_list.pkl')
	prediction_boosting(x,model_list, alpha_list,proba_threshold)

if __name__ == '__main__':
	main()