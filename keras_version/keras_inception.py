"""
Romain Gautron
"""

from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense
from keras import backend as k
img_width, img_height = 139, 139
epochs = 50

model = applications.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers:
    layer.trainable = False
    print(layer.name)
    print(layer.output)

#Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

# compile the model
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Train the model

model_final.fit(object, x = NULL, y = NULL, batch_size = NULL, epochs = epochs,
    verbose = getOption("keras.fit_verbose", default = 1), callbacks = NULL,
    view_metrics = getOption("keras.view_metrics", default = "auto"),
    validation_split = 0.9)
