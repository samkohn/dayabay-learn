'''
This module implements some convenient callback classes.

'''
from keras.callbacks import Callback
import h5py
import os.path

class WriteToLogger(Callback):
    def __init__(self, logger):
        self.logger = logger
        super(WriteToLogger, self).__init__()

    def on_epoch_end(self, epoch, logs):
        print ""
        self.logger.info('Ending epoch number %d' % epoch)
        self.logger.info('logs = %s', str(logs))

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

class SaveInputs(Callback):
    def __init__(self, data, folder):
        self.data = data
        self.folder = folder
        super(SaveInputs, self).__init__()

    def on_train_begin(self, logs=None):
        with h5py.File(os.path.join(self.folder, 'input.h5'), 'a') as f:
            f.create_dataset('input', data=self.data,
                    compression='gzip', chunks=True)
