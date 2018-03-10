"""
-----------------------------------------------
# @Author: Luc Blassel <lucblassel>
# @Date:   2018-01-15T12:04:51+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-17T13:21:02+01:00

parameter getter

-----------------------------------------------
"""
# import os
import json

def reader(filepath):
	'''
	Get parameters from a json file.
	Used in getArgs() in boosting.py.

	#Input
	filepath: Path of json file.

	#Output
	data: Dictionary of parameters with their labels.
	'''
    data = json.load(open(filepath))
    return data

def switchParams(data):
	'''
	Exchange parameters between two datasets with different labels.
	Not used currently.

	#Input
	data: Dictionary of parameters with their labels.

	#Output
	Nothing
	'''
    tmpTrain = data['trainLabels']
    tmpTest = data['testLabels']
    tmpNumtrain = data['trainNum']
    tmpNumtest = data['testNum']
    data["trainLabels"] = data["trainLabels2"]
    data["testLabels"] = data["testLabels2"]
    data["trainLabels2"] = tmpTrain
    data["testLabels2"] = tmpTest
    data["trainNum"] = data["trainNum2"]
    data["testNum"] = data["testNum2"]
    data["trainNum2"] = tmpNumtrain
    data["testNum2"] = tmpNumtest

def main():
    data = reader('config.json')
    print(data)
    switchParams(data)
    print(data)

if __name__ == "__main__":
    main()
