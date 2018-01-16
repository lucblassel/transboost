"""
-----------------------------------------------
# @Author: Luc Blassel <lucblassel>
# @Date:   2018-01-15T12:04:51+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-15T21:58:13+01:00

parameter getter

-----------------------------------------------
"""
# import os
import json

def reader(filepath):
    data = json.load(open(filepath))
    return data

def switchLabels(data):
    tmpTrain = data['trainLabels']
    tmpTest = data['testLabels']
    data["trainLabels"] = data["trainLabels2"]
    data["testLabels"] = data["testLabels2"]
    data["trainLabels2"] = tmpTrain
    data["testLabels2"] = tmpTest


def main():
    data = reader('config.json')
    print(data)
    switchLabels(data)
    print(data)

if __name__ == "__main__":
    main()
