import os

import numpy as np

import binvox_rw

ROOT = 'ModelNet10'
CLASSES = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']

X = {'train': [], 'test': []}
y = {'train': [], 'test': []}
for label, cl in enumerate(CLASSES):
    for split in ['train', 'test']:
        examples_dir = os.path.join('.', ROOT, cl, split)
        for example in os.listdir(examples_dir):
            if 'binvox' in example:
                with open(os.path.join(examples_dir, example), 'rb') as file:
                    data = np.int32(binvox_rw.read_as_3d_array(file).data)
                    padded_data = np.pad(data, 3, 'constant')
                    X[split].append(padded_data)
                    y[split].append(label)

np.savez_compressed('modelnet10.npz',
                    X_train=X['train'],
                    X_test=X['test'],
                    y_train=y['train'],
                    y_test=y['test'])