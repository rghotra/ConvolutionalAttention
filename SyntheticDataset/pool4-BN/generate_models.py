# IMPORTS --------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers, Model

import numpy as np
import requests as rq
import os, io, h5py

from bigbird.core import modeling
from tfomics import moana, evaluate
from tfomics.layers import MultiHeadAttention

import subprocess
import shlex

# RETRIVE DATASET ------------------------------------------------------

data = rq.get('https://www.dropbox.com/s/c3umbo5y13sqcfp/synthetic_dataset.h5?raw=true')
data.raise_for_status()

with h5py.File(io.BytesIO(data.content), 'r') as dataset:
    x_train = np.array(dataset['X_train']).astype(np.float32).transpose([0, 2, 1])
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32).transpose([0, 2, 1])
    y_valid = np.array(dataset['Y_valid']).astype(np.int32)
    x_test = np.array(dataset['X_test']).astype(np.float32).transpose([0, 2, 1])
    y_test = np.array(dataset['Y_test']).astype(np.int32)

# DEFINE MODELS --------------------------------------------------------

# change category, variants, names, old variant, new variant
category = 'activation'
variants = [True, False]

# places = len(str(max(variants)))
# names = [f"model-{str(variants[i]).zfill(places)}" for i in range(len(variants))]
names = ['model-relu', 'model-exp']

num_trials = 5

print(category, names)
if input("Continue? [y/n] ") != 'y':
	exit()

# BUILD MODELS --------------------------------------------------------

for i in range(len(variants)):

    if not os.path.exists(f'models/{category}/{names[i]}'): # model dir
        os.makedirs(f'models/{category}/{names[i]}')
    if not os.path.exists(f'motifs/{category}/{names[i]}'): # motif dir
        os.makedirs(f'motifs/{category}/{names[i]}')
    if not os.path.exists(f'results/{category}/{names[i]}'): # tomtom dir
    	os.makedirs(f'results/{category}/{names[i]}')
    if not os.path.exists(f'statistics/{category}/{names[i]}'): # statistics dir
    	os.makedirs(f'statistics/{category}/{names[i]}')

    for j in range(num_trials):
        # Input
        inputs = layers.Input(shape=(200, 4))

        # Convolutional Block
        nn = layers.Conv1D(filters=32, kernel_size=19, use_bias=False, padding='same')(inputs)
        nn = layers.BatchNormalization()(nn)
        if variants[i]:
            nn = layers.Activation('relu', name='conv_activation')(nn)
        else:
            nn = layers.Activation('exponential', name='conv_activation')(nn)
        nn = layers.MaxPool1D(pool_size=4)(nn)
        nn = layers.Dropout(0.1)(nn)

        # Multi-Head Attention
        nn, weights = MultiHeadAttention(num_heads=8, d_model=32)(nn, nn, nn)
        nn = layers.Dropout(0.1)(nn)

        nn = layers.Flatten()(nn)

        # Feed Forward
        nn = layers.Dense(512, use_bias=False)(nn)
        nn = layers.BatchNormalization()(nn)
        nn = layers.Activation('relu')(nn)
        nn = layers.Dropout(0.5)(nn)

        # Output
        outputs = layers.Dense(12, activation='sigmoid')(nn)

        # Compile model
        model = Model(inputs=inputs, outputs=outputs, name=names[i])
        print('\n' + model.name, 'trial-' + str(j+1))

        auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
        aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
        model.compile(tf.keras.optimizers.Adam(0.0005), loss='binary_crossentropy', metrics=[auroc, aupr])

        # Train Model
        lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_aupr', factor=0.2, patient=5, verbose=1, min_lr=1e-7, mode='max')
        model.fit(x_train, y_train, epochs=75, validation_data=(x_valid, y_valid), callbacks=[lr_decay], verbose=2, shuffle=True)

        # Save Model
        save_path = os.path.join('models', category, names[i], f'trial-{j+1}.h5')
        model.save_weights(save_path)

        # Extract PPMs
        index = [type(j) for j in model.layers].index(tf.keras.layers.Activation)

        ppms = moana.filter_activations(x_test, model, layer=index, window=20, threshold=0.5)
        ppms = moana.clip_filters(ppms, threshold=0.5, pad=3)

        motif_dir = f'motifs/{category}/{names[i]}/trial-{j+1}.txt'
        moana.meme_generate(ppms, output_file=motif_dir, prefix='filter')

        # Tomtom Comparison
        tomtom_dir = f'results/{category}/{names[i]}/trial-{j+1}'
        subprocess.call(shlex.split(f'./MotifComparison.sh {motif_dir} {tomtom_dir}'))

        # Analysis
        model_name = f'{names[i]}--trial-{j+1}'

        num_filters = moana.count_meme_entries(motif_dir)
        model_match_frac, match_any, filter_matches, filter_qvalues, motif_qvalues, hit_counts = evaluate.motif_comparison_synthetic_dataset(os.path.join(tomtom_dir, 'tomtom.tsv'), num_filters=num_filters)
        model_false_frac = match_any - model_match_frac

        results = model.evaluate(x_test, y_test, verbose=2)
        model_loss = results[0]
        model_auroc = results[1]
        model_aupr = results[2]

        statistics = np.array([model_name, model_loss, model_auroc, model_aupr, model_match_frac, model_false_frac])
        np.save(os.path.join('statistics', category, names[i], f'trial-{j+1}.npy'), statistics, allow_pickle=True)
