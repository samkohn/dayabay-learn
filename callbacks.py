'''
This module implements some convenient callback classes.

'''
from keras.callbacks import Callback
import h5py
import numpy as np
import os.path
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

class WriteToLogger(Callback):
    def __init__(self, logger):
        self.logger = logger
        super(WriteToLogger, self).__init__()

    def on_epoch_end(self, epoch, logs):
        print ""
        self.logger.info('Ending epoch number %d' % epoch)

class SaveIntermediateResults(Callback):
    def __init__(self, data, models_to_predict, folder, every_x_epochs):
        self.data = data
        self.models = models_to_predict
        self.folder = folder
        self.interval = every_x_epochs
        super(SaveIntermediateResults, self).__init__()

    def on_epoch_end(self, epoch, logs):
        if epoch % self.interval != self.interval-1:
            return
        predictions = {}
        for name, model in self.models.iteritems():
            predictions[name] = model.predict(self.data)
        with h5py.File(os.path.join(self.folder, 'intermediate_%d.h5' % epoch)) as f:
            for name, model in self.models.iteritems():
                dataset = f.create_dataset(name,
                    data=predictions[name],
                    compression='gzip', chunks=True)
        self.model.save_weights(os.path.join(self.folder, 'weights_%d.h5' % epoch))

    def on_train_end(self, logs):
        predictions = {}
        for name, model in self.models.iteritems():
            predictions[name] = model.predict(self.data)
        with h5py.File(os.path.join(self.folder, 'results.h5'), 'a') as f:
            for name, model in self.models.iteritems():
                dataset = f.create_dataset(name,
                    data=predictions[name],
                    compression='gzip', chunks=True)
        self.model.save_weights(os.path.join(self.folder, 'weights.h5'))

class SaveCostCurve(Callback):
    def __init__(self, folder, every_x_epochs):
        self.interval = every_x_epochs
        self.folder = folder

    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        if epoch % self.interval != self.interval-1:
            return
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (mean squared error)')
        plt.title('Cost curve')
        plt.savefig(os.path.join(self.folder, 'cost.pdf'))
        plt.clf()

    def on_train_end(self, logs):
        with h5py.File(os.path.join(self.folder, 'results.h5'), 'a') as f:
            f.create_dataset('cost', data=self.losses,
                    compression='gzip', chunks=True)

class SaveInputs(Callback):
    def __init__(self, data, classes, folder):
        self.data = data
        self.classes = classes
        self.folder = folder
        super(SaveInputs, self).__init__()

    def on_train_begin(self, logs=None):
        with h5py.File(os.path.join(self.folder, 'input.h5'), 'a') as f:
            f.create_dataset('input', data=self.data,
                    compression='gzip', chunks=True)
            f.create_dataset('labels', data=self.classes,
                    compression='gzip', chunks=True)
