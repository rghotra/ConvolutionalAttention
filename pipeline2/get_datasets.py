import numpy as np
import requests as rq
import io, h5py


# Synthetic Dataset

data = rq.get('https://www.dropbox.com/s/c3umbo5y13sqcfp/synthetic_dataset.h5?raw=true')
data.raise_for_status()

with h5py.File(io.BytesIO(data.content), 'r') as dataset:
    x_train = np.array(dataset['X_train']).astype(np.float32).transpose([0, 2, 1])
    y_train = np.array(dataset['Y_train']).astype(np.float32)
    x_valid = np.array(dataset['X_valid']).astype(np.float32).transpose([0, 2, 1])
    y_valid = np.array(dataset['Y_valid']).astype(np.int32)
    x_test = np.array(dataset['X_test']).astype(np.float32).transpose([0, 2, 1])
    y_test = np.array(dataset['Y_test']).astype(np.int32)

synthetic_dataset = np.array([x_train, y_train, x_valid, y_valid, x_test, y_test])
np.save('synthetic_dataset.npy', synthetic_dataset, allow_pickle=True)
print('saved synthetic dataset')


# Synthetic Coded Dataset

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

synthetic_coded_dataset = np.array([x_train, y_train, x_valid, y_valid, x_test, y_test, model_test])
np.save('synthetic_coded_dataset.npy', synthetic_coded_dataset, allow_pickle=True)
print('saved synthetic coded dataset')


# Synthetic Correlated Dataset





















