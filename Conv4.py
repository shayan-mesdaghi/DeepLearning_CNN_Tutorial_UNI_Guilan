# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 17:22:24 2022

@author: SHM
"""

from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
import datetime#1
from keras.models import Model
from keras import layers
import keras

from plot_history import plot_history #Function ploting
# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data attributes
m = train_images.shape[1]
n = test_images.shape[1]
X_train = train_images.reshape(len(train_images), m, n, 1)/255# in conv input is size dataset (number of data, n, m, channel (RGB))
X_test  = test_images.reshape(len(test_images),m ,n , 1)/255
#X_test = test_images.reshape(10000, 28, 28, 1)

# X_train = X_train.astype('float32')/255
# X_test = X_test.astype('float32')/255

from keras.utils import np_utils
Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)

#==================================================
# Creating our model

myInput = layers.Input(shape=(m, n, 1))
conv1 = layers.Conv2D(16, 3, activation='relu', padding='same', strides=2)(myInput)
conv2 = layers.Conv2D(32, 3, activation='relu', padding='same', strides=2)(conv1)
flat = layers.Flatten()(conv2)
out_layer = layers.Dense(10, activation='softmax')(flat)

myModel = Model(myInput, out_layer)

myModel.summary()
myModel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.utils import plot_model
plot_model(myModel, to_file='CNN2_model.png', show_shapes=True)#pdf
#==================================================
# Train our model
start = datetime.datetime.now()
hist = myModel.fit(X_train, Y_train, batch_size=128, epochs=2, validation_split=0.2)
end = datetime.datetime.now()
elapsed = end-start
print('Total training time: ', str(elapsed))
plot_history(hist)
# Evaluation
test_loss, test_acc = myModel.evaluate(X_test, Y_test)
test_labels_p = myModel.predict(X_test)
test_labels_p = np.argmax(test_labels_p, axis=1)
print("Test_Accuracy: {:.2f}%".format(myModel.evaluate(np.array(X_test), np.array(Y_test))[1]*100))#2

