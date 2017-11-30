# -*- coding: utf-8 -*-
# @Author: Romain
# @Date:   2017-11-30 15:05:11
# @Last Modified by:   Romain
# @Last Modified time: 2017-11-30 17:42:16
import numpy as np
def class_binariser(array):
	"""
	this functions turns for example [2,3,3,3,2] in [0,1,1,1,0]
	"""
	to_return = []
	max_value = np.max(array)
	for element in array:
		if element == max_value:
			to_return.append(1)
		else:
			to_return.append(0)
	return np.array(to_return)

def label_binariser(array):
	"""
	this functions turns for example [[0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0]] in [[0,1],[0,1],[1,0]]
	"""
	to_return = []
	unique_vectors = np.vstack({tuple(row) for row in array})
	nonzeroind0 = np.nonzero(unique_vectors[0])[0][0]
	nonzeroind1 = np.nonzero(unique_vectors[1])[0][0]
	max_zero = np.max([nonzeroind0,nonzeroind1])#index of the first non zero_element
	for sub_array in array :
		if np.nonzero(sub_array)[0][0] == max_zero:
			to_return.append([0,1])
		else:
			to_return.append([1,0])
	return np.array(to_return)

def class_name_binariser(class_names,cls):
	"""
	this function selects only the needed class_names
	"""
	cls = np.unique(cls)
	max_index_class = np.max(cls)
	min_index_class = np.min(cls)
	return list(class_names[i] for i in [min_index_class,max_index_class])

def main():
	"""
	this function stands for testing purpose
	"""
	to_print0 = class_binariser(np.array([3,3,3,3,4,4,4,3,3,4]))
	print(to_print0)
	to_print1 = label_binariser(np.array([[0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,0,1,0,0]]))
	print(to_print1)

if __name__ == '__main__':
	main()