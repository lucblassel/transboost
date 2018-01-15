# -*- coding: utf-8 -*-
# @Author: romaingautronapt
# @Date:   2018-01-15 14:59:20
# @Last Modified by:   romaingautronapt
# @Last Modified time: 2018-01-15 17:45:33
from keras_inception import *
import numpy as np
import time


def take(tab,indexes):
	output = np.zeros(tab.shape)

	c=0
	for i in indexes:
		output[c] = tab[i]
		c+=1
	return output

def booster(full_model,times,x_train,y_train_bin,epochs,threshold,layer_limit):
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

		current_model = first_layers_modified_model_builder(full_model,layer_limit)
		error = 0
		while error == 1 or error == 0 :
			current_model = first_layers_modified_model_builder(full_model,layer_limit)
			first_layers_modified_model_trainer(current_model,x_train_boost,y_train_bin,epochs,threshold)
			error = 1 - current_model.evaluate(x_train, y_train_bin, verbose=1)[1]
		alpha = .5*np.log((1-error)/error)

		error_list.append(error)
		model_list.append(current_model)
		alpha_list.append(alpha)

		for i in range(train_length):
			predicted_prob = current_model.predict(x_train_boost)
			print("index ",np.where(predicted_prob[i] == predicted_prob[i].max())[0][0],"predicted_prob[i] ",predicted_prob[i])
			print("index ",np.where(y_train_bin[i] == 1)[0][0],"y_train_bin[i] ",y_train_bin[i])
			if np.where(predicted_prob[i] == predicted_prob[i].max())[0][0] == np.where(y_train_bin[i] == 1)[0][0]:
				prob[i] = prob[i]*np.exp(-alpha)
			else:
				prob[i] = prob[i]*np.exp(alpha)
		prob = prob / np.sum(prob)
		break
	return model_list, error_list, alpha_list

def main():
	""" this function stands for testing purposes
	"""
	wantedLabels=['dog','truck']
	trainnum,testnum = 100,100
	x_train, y_train_bin, x_test, y_test_bin = loader(wantedLabels,trainnum,testnum)
	img_width,img_height = 160,160
	full_model = full_model_builder(img_width,img_height)
	layer_limit = 10
	epochs = 1
	threshold = .5
	model_list, error_list, alpha_list = booster(full_model,5,x_train,y_train_bin,epochs,threshold,layer_limit)
	print("model_list ", model_list)
	print("error_list ", error_list)
	print("alpha_list ", alpha_list)

if __name__ == '__main__':
	main()
