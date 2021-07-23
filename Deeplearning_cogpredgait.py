# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from __future__ import absolute_import, division, print_function

import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

column_names = ['moca_tot',	'speed_init_L', 'speed_init_R', 'speed_fast_L', 'speed_fast_R', 'speed_dual_L', 'speed_dual_R']
raw_dataset = pd.read_excel('thesis_speed_variables_prediction_MoCA.xlsx', names = column_names)
dataset = raw_dataset.copy()

dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8, random_state=1)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('moca_tot')
test_labels = test_dataset.pop('moca_tot')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def stand(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = stand(train_dataset)
normed_test_data = stand(test_dataset)

def build_model():
    model = keras.Sequential([layers.Dense(12, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
                             layers.Dense(8, activation=tf.nn.relu),
                             layers.Dense(6, activation=tf.nn.relu),
                             layers.Dense(3, activation=tf.nn.relu),
                             layers.Dense(1)
                             ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.0001)
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    
    return model

model = build_model()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end=' ')
    
EPOCHS = 10000
    
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience = 50)

history = model.fit(
         normed_train_data, train_labels, 
         epochs= EPOCHS, validation_split = 0.2, verbose = 0, 
         callbacks=[early_stop, PrintDot()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean absolute error')
    plt.plot(hist['epoch'], hist['mae'], label = 'Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')
    plt.savefig('deep_learning_training.png')
    plt.legend()
    
    #plt.figure()
    #plt.xlabel('Epoch')
    #plt.ylabel('Mean square error (MoCa)')
    #plt.plot(hist['epoch'], hist['mse'], label = 'Train Error')
    #plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')
    #plt.legend()

plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose = 0) 

print(mae)
test_predictions = model.predict(normed_test_data).flatten()

model.summary()

plt.scatter(test_labels, test_predictions)
print(test_labels)
print(test_predictions)
plt.xlabel("True values MoCA")
plt.ylabel("Predicted MoCA")
_ = plt.plot([20, 35], [20, 35], color='grey')






