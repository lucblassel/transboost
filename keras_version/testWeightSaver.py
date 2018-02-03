from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Activation
from keras import backend as k
from keras import applications
from keras import optimizers
import copy as cp

def small_net_builder(originalSize,resizeFactor,lr):
        """
        romain.gautron@agroparistech.fr
        """
        img_size = originalSize*resizeFactor

        if k.image_data_format() == 'channels_first':
                input_shape = (3, img_size, img_size)
        else:
                input_shape = (img_size, img_size, 3)

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(34))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(optimizer = optimizers.Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])

        return model


def trainedWeightSaver(model,layerLimit,modelName):
	"""
	luc blassel
	saves weights of modified layers
	"""
	model_copy = Sequential()
	for layer in model.layers[:layerLimit]:
		model_copy.add(layer)

	model_copy.save_weights(modelName)
	del model_copy

def main():
	originalSize = 32
	resizeFactor = 5
	lr = 0.0001
	model = small_net_builder(originalSize,resizeFactor,lr)
	trainedWeightSaver(model,5,'testModel')
	
if __name__=='__main__':
	main()
