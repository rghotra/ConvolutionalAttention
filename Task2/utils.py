from tfomics import explain, evaluate

import numpy as np
import requests as rq
import io, h5py

def get_synthetic_coded_dataset():

    data = rq.get('https://www.dropbox.com/s/5iww0ootxkr6e21/synthetic_code_dataset.h5?raw=true')
    data.raise_for_status()

    with h5py.File(io.BytesIO(data.content), 'r') as dataset:
        x_train = np.array(dataset['X_train']).astype(np.float32).transpose([0, 2, 1])
        y_train = np.array(dataset['Y_train']).astype(np.float32)
        x_valid = np.array(dataset['X_valid']).astype(np.float32).transpose([0, 2, 1])
        y_valid = np.array(dataset['Y_valid']).astype(np.int32)
        x_test = np.array(dataset['X_test']).astype(np.float32).transpose([0, 2, 1])
        y_test = np.array(dataset['Y_test']).astype(np.int32)
        model_test = np.array(dataset['model_test']).astype(np.float32).transpose([0, 2, 1])

    return x_train, y_train, x_valid, y_valid, x_test, y_test, model_test

def get_statistics(model, x_test, y_test, model_test, num_analyze=500, threshold=0.1, top_k=10):

    # get positive label sequences and sequence model
    pos_index = np.where(y_test[:,0] == 1)[0]   
    X = x_test[pos_index[:num_analyze]]
    X_model = model_test[pos_index[:num_analyze]]

    # instantiate explainer class
    explainer = explain.Explainer(model, class_index=0)

    # calculate attribution maps
    saliency_scores = explainer.saliency_maps(X)

    # reduce attribution maps to 1D scores
    sal_scores = explain.grad_times_input(X, saliency_scores)
    
    saliency_roc, saliency_pr = evaluate.interpretability_performance(sal_scores, X_model, threshold)
    sal_signal, sal_noise_max, sal_noise_mean, sal_noise_topk = evaluate.signal_noise_stats(sal_scores, X_model, top_k, threshold)
    snr = evaluate.calculate_snr(sal_signal, sal_noise_topk)
    
    results = model.evaluate(x_test, y_test, verbose=2)
    model_loss = results[0]
    model_auroc = results[1]
    model_aupr = results[2]
    
    return saliency_roc, saliency_pr, snr, model_aupr, model_auroc
