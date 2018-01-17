"""
# @Author: Luc Blassel <zlanderous>
# @Date:   2018-01-15T00:21:20+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-17T13:38:25+01:00

Romain Gautron
"""
from binariser import *
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense
from keras import backend as k
from callbackBoosting import *
import paramGetter as pg
import sys
from boosting import *
import time
from scipy.misc import imshow

def getArgs():
    """
    gets the parameters from config file
    """
    if len(sys.argv)==1:
        print("please specify path of config file...")
        sys.exit()

    path = sys.argv[1]
    return pg.reader(path) #dictionary with relevant parameters

def show5(set):
    """
    shows 5 images from set
    """
    for i in range(5):
        imshow(set[i])



############################################################################
# MAIN                                                                        #
############################################################################

def main():
    """
    this function stands for testing purposes
    """

    print('getting parameters')
    params = getArgs()
    print('reading data')
    x_train, y_train_bin, x_test, y_test_bin = loader(**params)



    print("data loaded")
    full_model = full_model_builder(**params)
    print("full model built")
    
    show5(x_train)
    show5(x_test)

    #score = full_model_trainer(full_model,x_train,y_train_bin,x_test,y_test_bin,**params)
    print("modified model trained")

    # print("full model score ",score)
    # modified_model = first_layers_modified_model_builder(full_model,**params)
    # print("modified model built")
    # first_layers_modified_model_trainer(modified_model,x_train,y_train_bin,**params)
    # print("modified model trained")

    #switching parameters for boosting
    pg.switchParams(params)
    x_train, y_train_bin, x_test, y_test_bin = loader(**params)

    show5(x_train)
    show5(x_test)

    time.sleep(30)
    # Boosting
    model_list, error_list, alpha_list = booster(full_model,x_train,y_train_bin,**params)
    print("model_list ", model_list)
    print("error_list ", error_list)
    print("alpha_list ", alpha_list)
    y_pred = prediction_boosting(x_test,model_list,error_list)
    print(accuracy(y_test,y_pred))
if __name__ == '__main__':
    main()
