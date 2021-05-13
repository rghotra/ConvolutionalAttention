from tfomics import moana, evaluate

import numpy as np
import requests as rq
import io, h5py

import subprocess
import shlex

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
    index = [i.name for i in model.layers].index('conv_activation')
    
    ppms = moana.filter_activations(x_test, model, layer=index, window=20,threshold=0.5)
    ppms = moana.clip_filters(ppms, threshold=0.5, pad=3)
    
    return ppms

def tomtom(motif_dir, output_dir):
    subprocess.call(shlex.split('chmod +x ./motif_comparison.sh'))
    subprocess.call(shlex.split(f'./motif_comparison.sh {motif_dir} {output_dir}'))

def analysis(name, motif_dir, tomtom_dir, model, x_test, y_test):
    model_name = name

    num_filters = moana.count_meme_entries(motif_dir)
    model_match_frac, match_any, filter_matches, filter_qvalues, motif_qvalues, hit_counts = evaluate.motif_comparison_synthetic_dataset(tomtom_dir + '/tomtom.tsv', num_filters=num_filters)
    model_false_frac = match_any - model_match_frac

    results = model.evaluate(x_test, y_test, verbose=2)
    model_loss = results[0]
    model_auroc = results[1]
    model_aupr = results[2]

    statistics = np.array([model_name, model_loss, model_auroc, model_aupr, model_match_frac, model_false_frac])
    return statistics
    