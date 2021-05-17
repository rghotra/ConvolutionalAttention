# Imports
import tensorflow as tf

import os
import numpy as np
from six.moves import cPickle

import models
import utils

# Retrieve dataset
x_train, y_train, x_valid, y_valid, x_test, y_test, model_test = utils.get_synthetic_coded_dataset()

def execute_pipeline(baseline, category, trial, model, epochs=75):
    
    global x_train, y_train, x_valid, y_valid, x_test, y_test, model_test

    
    # Create directories
    model_dir = f'{baseline}/models/{category}'
    stats_dir = f'{baseline}/stats/{category}'
    logs_dir = f'{baseline}/history/{category}'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    model_dir += f'/trial-{trial}/weights'
    stats_dir += f'/trial-{trial}.npy'
    logs_dir += f'/trial-{trial}.pickle'


    # Train model
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    model.compile(tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=[auroc, aupr])

    lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_aupr', factor=0.2, patient=5, verbose=1, min_lr=1e-7, mode='max')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_aupr', patience=15, verbose=1, mode='max')
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_valid, y_valid), callbacks=[lr_decay, early_stop], verbose=1)

    model.save_weights(model_dir)
    
    with open(logs_dir, 'wb') as handle:
        cPickle.dump(history.history, handle)
        
        
    # Obtain saliency scores
    sal_roc, sal_pr, snr, model_aupr, model_auroc = utils.get_statistics(model, x_test, y_test, model_test)
    stats = np.array([sal_roc, sal_pr, snr, model_aupr, model_roc])
    np.save(stats_dir, stats, allow_pickle=True)