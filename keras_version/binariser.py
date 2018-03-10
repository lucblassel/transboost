# @Author: Luc Blassel
# @Date:   2018-01-15T10:46:32+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   Luc Blassel
# @Last modified time: 2018-01-15T11:08:54+01:00

"""
-----------------------------------------------
Binariser
-----------------------------------------------
"""

import dataProcessing
import numpy as np

def binarise(y):
    """
    Binarize the labels vectors (y) with one hot encoding. 
	#Input:
	y: the arrays of the labels of classes in a format of array without duplicated.
	#Outputs:
	A binarized vector in a format [[0,1], [1,0],[0,1]…]

    """
    unique = np.unique(y)
    if len(unique) >2:
        print("There are more than 2 classes, cannot binarize classes")
        return
    elif len(unique) ==1:
        print("There is only one class, cannot binarize")
        return
    elif len(unique) ==0:
        print("empty array...")
        return
    else:
        binarised = np.zeros((y.shape[0],2),dtype=int)

        for i in range(len(y)):
            if y[i] == unique[0]:
                binarised[i] = np.array([1,0])
            else:
                binarised[i] = np.array([0,1])
    return binarised

def main():
	"""
	Binarize  y_train set with one hot code.
	#Input:
	No Input. 
	#Output:
	Print the original array and binarized array of y_train. 
	"""
    labels = ['dog','truck']
    trainNum = 10

    train,test = dataProcessing.loadRawData()
    x_train,y_train = dataProcessing.loadTrainingData(train,labels,trainNum)
    bina = binarise(y_train)

    for i in range(len(bina)):
        print(y_train[i],bina[i])

if __name__ == "__main__":
    main()
