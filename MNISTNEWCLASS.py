#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 16:13:21 2020

@author: elijahchandler
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 18:48:41 2020
@author: elijahchandler
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

#(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

raw_test_df = pd.read_csv('/Users/elijahchandler/Desktop/digit-recognizer/test.csv')
raw_train_df = pd.read_csv('/Users/elijahchandler/Desktop/digit-recognizer/train.csv')

x_train_set = raw_train_df.drop(['label'], axis=1)
y_train_set = raw_train_df['label']
x_train_set = x_train_set.values
y_train_set = y_train_set.values

X_train, X_val, Y_train, Y_val = train_test_split(x_train_set, y_train_set, test_size=0.20)
#%%
from tensorflow.keras.utils import to_categorical
Y_train = to_categorical(Y_train)
Y_val = to_categorical(Y_val)
#%%
#do i need to keep this numVal = 1000

#numVal = 1000
#X_val = X_train[:numVal]
##Y_val = Y_train[0:numVal,:]
#X_train = X_train[numVal:]
#Y_train = Y_train[0:numVal,:]
#%%

model = tf.keras.models.Sequential([
    ##does the shape here need to change?
    tf.keras.layers.Dense(64, activation = 'relu', input_shape = (784,)),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
    ])

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#%%
numEpochs = 9
history = model.fit(X_train, Y_train, epochs = numEpochs, batch_size = 512, validation_data = (X_val, Y_val))
history_dict = history.history
#%%

training_loss = history_dict['loss']
validation_loss = history_dict['val_loss']
training_accuracy = history_dict['accuracy']
validation_accuracy = history_dict['val_accuracy']
epochs = np.arange(1,numEpochs+1)

#plotting

loss, accuracy = model.evaluate(X_val, Y_val)
#%%
pred = model.predict(raw_test_df.values)
pred = np.argmax(pred, axis = 1)
#use the sample submission and use their form
#%%
df = pd.DataFrame()
df['ImageID'] = np.arange(28000)+1
df['label'] = pred
#%%
df.to_csv('/Users/elijahchandler/Desktop/TFMNISTFINAL',index = False)
