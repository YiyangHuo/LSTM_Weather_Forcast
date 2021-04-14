#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 4/13/21 9:55 PM
#@Author: Yiyang Huo
#@File  : train.py.py

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

TRAIN_SPLIT = 40000
BATCH_SIZE = 256
BUFFER_SIZE = 10000
EPOCHS = 10
EVALUATION_INTERVAL = 300
TARGET_INDEX = 2
future_target = 72
feature_output = 2
STEP = 2
past_history = 240

added_features = ["Precip", "Sol_Rad", "Air_Temp", "Wind_Speed", "Vap_Pres", "Rel_Hum"]


def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, TARGET_INDEX]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()

def create_time_steps(length):
    return list(range(-length, 0))

def create_model(shape):
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(32, return_sequences=True,input_shape=shape))
    model.add(keras.layers.LSTM(16, activation='relu'))
    #model.add(keras.layers.RepeatVector(future_target))
    model.add(keras.layers.Dense(future_target))
    return model

def preprocessdata(data, name):
    newset = data[name].values

    uni_train_mean = newset[:TRAIN_SPLIT].mean()
    uni_train_std = newset[:TRAIN_SPLIT].std()
    newset = (newset - uni_train_mean)/uni_train_std
    return newset


def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)  # step表示滑动步长
        data.append(dataset[indices])

        #labels.append(np.mean(target[i:i + target_size], axis=1))#.flatten('F'))
        labels.append(target[i:i + target_size])  # .flatten('F'))

    return np.array(data), np.array(labels)


def train():
    data = pd.read_csv("./data/hourly2.csv").dropna()
    tf.random.set_seed(233)

    train_data = preprocessdata(data, added_features)

    xtrain, ytrain = multivariate_data(train_data, train_data[:, TARGET_INDEX], 0, TRAIN_SPLIT, past_history, future_target, STEP,
                                       single_step=False)
    xval, yval = multivariate_data(train_data, train_data[:, TARGET_INDEX], TRAIN_SPLIT, None, past_history, future_target, STEP,
                                   single_step=False)
    train_data_multi = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
    train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data_multi = tf.data.Dataset.from_tensor_slices((xval, yval))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    multi_step_model = create_model(xtrain.shape[-2:])

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL)
    # validation_data=val_data_multi,
    # validation_steps=50)
    multi_step_model.save("model.h5")

    for x, y in val_data_multi.take(3):
        multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
    loss = multi_step_model.evaluate(xval, yval, verbose=1)

def test():
    model = keras.models.load_model("model.h5")
    data = pd.read_csv("./data/hourly2.csv").fillna(value=0)
    tf.random.set_seed(233)

    train_data = preprocessdata(data, added_features)
    xval, yval = multivariate_data(train_data, train_data[:, TARGET_INDEX], TRAIN_SPLIT, None, past_history, future_target, STEP,
                                   single_step=False)

    val_data_multi = tf.data.Dataset.from_tensor_slices((xval, yval))
    val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

    for x, y in val_data_multi.take(3):
        multi_step_plot(x[0], y[0], model.predict(x)[0])
    loss = model.evaluate(xval, yval, verbose=1)

if __name__ == "__main__":
   #train()
   test()