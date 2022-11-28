# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:31:17 2022

@author: SHM
"""

from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
import datetime#1
from keras.models import Model #Instead of the library Sequential
from keras import layers #Instead of the library Conv2D, MAXPool2D, Flatten, Dense,....
import keras #Instead of the library loss function
# Load data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Data attributes
m = train_images.shape[1]
n = test_images.shape[1]

X_train = train_images.reshape(len(train_images), m, n, 1)/255# in conv input is size dataset (number of data, n, m, channel (RGB))
X_test  = test_images.reshape(len(test_images), m, n, 1)/255
#X_test = test_images.reshape(10000, 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

from keras.utils import np_utils
Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)

#==================================================
# Creating our model

myInput = layers.Input(shape=(m, n, 1))
conv1 = layers.Conv2D(16, 3, activation = 'relu', padding = 'same')(myInput)#(Num # filters, Window filter(w,w), activation("valid"` or `"same"`), strides(defult=1))
pool1 = layers.MaxPooling2D((2, 2))(conv1)#(window(w,w), pading(defult))
conv2 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same')(pool1)
pool2 = layers.MaxPooling2D((2, 2))(conv2) # we like a classifier but out put is picture
flat = layers.Flatten()(pool2) # flatten layer change every inputs data to vectors
out_layer = layers.Dense(10, activation='softmax')(flat) #last layer is classifier and use layer dense

myModel = Model(myInput, out_layer) #in functional API you should make a Model with input net and output net

myModel.summary()
myModel.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy']) #keras.___ and keras.losses.___

from keras.utils import plot_model
plot_model(myModel, to_file='CNN_model.png', show_shapes=True)#pdf
#==================================================
# Train our model
start = datetime.datetime.now()
hist = myModel.fit(X_train, Y_train, batch_size=128, epochs=1, validation_split=0.2)
end = datetime.datetime.now()
elapsed = end-start
print('Total training time: ', str(elapsed))

# Evaluation
test_loss, test_acc = myModel.evaluate(X_test, Y_test)
test_labels_p = myModel.predict(X_test)
test_labels_p = np.argmax(test_labels_p, axis=1)
print("Test_Accuracy: {:.2f}%".format(myModel.evaluate(np.array(X_test), np.array(Y_test))[1]*100))#2

#==================================================
#Visual
history = hist.history

losses = history['loss']
val_losses = history['val_loss']
accuracies = history['accuracy']
val_accuracies = history['val_accuracy']

plt.figure()  
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses, 'red')
plt.grid()
plt.plot(val_losses, 'blue')
plt.legend(['loss', 'val_loss'])
    
plt.figure()
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.grid()
plt.plot(accuracies)
plt.plot(val_accuracies)
plt.legend(['accuracy', 'val_accuracy'])