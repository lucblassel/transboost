"""
# @Author: Luc Blassel <zlanderous>
# @Date:   2018-01-15T00:21:20+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-17T13:24:32+01:00

Romain Gautron
"""
from dataProcessing import *
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
#####################
# LOADING DATA        #
#####################

def loader(trainLabels,testLabels,trainNum,testNum,**kwargs):
    """
    this function loads the datasets from CIFAR 10 with correct output in order to feed inception
    OUTPUT : 32*resizefactor,32*resizefactor
    """

    raw_train,raw_test = loadRawData()
    x_train,y_train = loadTrainingData(raw_train,labels=trainLabels,trainCases=trainNum)
    # x_train,y_train = loadTrainingData(raw_train,trLabels,trNum)
    x_test,y_test = loadTestingData(raw_test,labels=testLabels,testCases=testNum)
    # x_test,y_test = loadTestingData(raw_test,teLabels,teNum)
    y_train_bin,y_test_bin = binarise(y_train),binarise(y_test)

    return x_train, y_train_bin, x_test, y_test_bin

#####################################
# BUILDING MODEL FOR TWO CLASSES    #
#####################################

def full_model_builder(originalSize,resizeFactor,**kwargs):
    """
    this function builds a model that outputs binary classes
    INPUTS :
    - img_width >=139
    - img_height >=139
    OUTPUTS :
    -full model
    """
    img_width = originalSize*resizeFactor
    img_height = originalSize*resizeFactor

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
# TRAINING AND TESTING FULL MODEL    #
#####################################

def full_model_trainer(model,x_train,y_train_bin,x_test,y_test_bin,epochs,**kwargs):
    """
    this function's purpose is to train the full model
    INPUTS : the model to train
    OUPUTS : the model score
    """
    model.fit(x = x_train, y = y_train_bin, batch_size = 10, epochs = epochs,validation_split = 0.1)
    score = model.evaluate(x_test, y_test_bin, verbose=1)
    return score


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
    model.fit(x = x_train, y = y_train_bin, batch_size = 10, epochs = epochs,validation_split = 0.1,callbacks = [callbackBoosting(threshold)])


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

    show5(x_train)
    show5(x_test)

    print("data loaded")
    full_model = full_model_builder(**params)
    print("full model built")

    # params["testLabels"] = para
    # score = full_model_trainer(full_model,x_train,y_train_bin,x_test,y_test_bin,**params)
    # print("modified model trained")
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
