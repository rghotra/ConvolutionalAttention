import tensorflow as tf
from tensorflow.keras import layers, Model
from tfomics import moana

import numpy as np
import requests as rq
import os, io, h5py

import subprocess
import shlex

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

def get_ppms(model, x_test):
    index = [type(i) for i in model.layers].index(tf.keras.layers.Activation)
    
    ppms = moana.filter_activations(x_test, model, layer=index, window=20,threshold=0.5)
    ppms = moana.clip_filters(ppms, threshold=0.5, pad=3)
    
    return ppms

def tomtom(motif_dir, output_dir):
    subprocess.call(shlex.split(f'./motif_comparison.sh {motif_dir} {output_dir}'))
