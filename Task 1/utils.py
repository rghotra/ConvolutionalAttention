import tensorflow as tf
from tensorflow.keras import layers, Model

import numpy as np
import requests as rq
import os, io, h5py

import models

tf.config.run_functions_eagerly(True)

def get_synthetic_dataset():

    data = rq.get('https://www.dropbox.com/s/c3umbo5y13sqcfp/synthetic_dataset.h5?raw=true')
    data.raise_for_status()

    with h5py.File(io.BytesIO(data.content), 'r') as dataset:
        x_train = np.array(dataset['X_train']).astype(np.float32).transpose([0, 2, 1])
        y_train = np.array(dataset['Y_train']).astype(np.float32)
        x_valid = np.array(dataset['X_valid']).astype(np.float32).transpose([0, 2, 1])
        y_valid = np.array(dataset['Y_valid']).astype(np.int32)
        x_test = np.array(dataset['X_test']).astype(np.float32).transpose([0, 2, 1])
        y_test = np.array(dataset['Y_test']).astype(np.int32)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def train_model(model, train, valid, epochs=75, seed=None):
    if seed is not None:
        tf.random.set_seed(seed)
    
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    model.compile(tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=[auroc, aupr])

    lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_aupr', factor=0.2, patient=5, verbose=1, min_lr=1e-7, mode='max')
    model.fit(train[0], train[1], epochs=epochs, validation_data=(valid[0], valid[1]), callbacks=[lr_decay], verbose=1)
    
    
