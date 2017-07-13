"""
A module for creating a t-SNE embedding given a list of vectors.

"""

import h5py
import numpy as np
from sklearn.manifold import TSNE

resultsfilename = 'batch/tmp_output/e1000_w16_n18000_keras/results.h5'
labelsfilename = 'batch/tmp_output/e1000_w16_n18000_keras/results.h5'
outfilename = 'batch/tmp_output/e1000_w16_n18000_keras/tsne.h5'
num_events = 5000
dataset_name = 'encodings'
classes = range(2)
numclasses = len(classes)
def vector_transform_function(dataset4d):
    return dataset4d[:, :, 0, 0]
with h5py.File(resultsfilename, 'r') as infile:
    dataset_raw = infile[dataset_name]
    dataset = vector_transform_function(dataset_raw)
with h5py.File(labelsfilename, 'r') as infile:
    labels = infile['labels'][:]
# Assume that the first half of the dataset is type 1 and the second half is
# type 2
dataset_split = {}
for label in classes:
    dataset_split[label] = dataset[labels==label]
dataset = np.vstack([dataset_split[x][:num_events/numclasses] for x in
    classes])
myTSNE = TSNE(random_state=0)
result = myTSNE.fit_transform(dataset)
with h5py.File(outfilename, 'a') as outfile:
    outfile.create_dataset('tsne', data=result, compression='gzip',
            chunks=True)
