"""
# @Author: Luc Blassel <zlanderous>
# @Date:   2018-01-15T00:21:20+01:00
# @Email:  luc.blassel@agroparistech.fr
# @Last modified by:   zlanderous
# @Last modified time: 2018-01-15T14:31:14+01:00

Romain Gautron
"""
import os
os.chdir("/Users/Romain/Documents/Cours/APT/IODAA/transboost/keras_version")
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

trainnum = 1000
testnum = 1000
wantedLabels = ['dog','truck']
img_width, img_height = resizeFactor*32,resizeFactor*32
epochs = 5
threshold = .8

#####################
# LOADING DATA		#
#####################

raw_train,raw_test = loadRawData()
x_train,y_train = loadTrainingData(raw_train,wantedLabels,trainnum)
x_test,y_test = loadTestingData(raw_test,wantedLabels,testnum)
y_train_bin,y_test_bin = binarise(y_train),binarise(y_test)

#####################
# BUILDING MODEL	#
#####################


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

#####################
# TRAINING MODEL	#
#####################

model_final.fit(x = x_train, y = y_train_bin, batch_size = 10, epochs = epochs,validation_split = 0.1,callbacks = [callbackBoosting(threshold)])

#####################
# TESTING MODEL		#
#####################

score = model_final.evaluate(x_test, y_test_bin, verbose=1)

print(score)


############################################################################
# TRAINING FIRST LAYERS
############################################################################

model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

for layer in model.layers[:4]:
    layer.trainable = True
    previous_weights = layer.get_weights()
    new_weights = list((10*np.random.random((np.array(previous_weights).shape))))
    layer.set_weights(new_weights)

for layer in model.layers[4:]:
    layer.trainable = False


model_final.fit(x = x_train, y = y_train_bin, batch_size = 10, epochs = epochs,validation_split = 0.1,callbacks = [callbackBoosting(threshold)])
