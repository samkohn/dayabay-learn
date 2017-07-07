import numpy as np
import h5py
import sys
import networks.BasicConvAE as nn
import networks.preprocessing as preprocessing
from util.data_loaders import get_ibd_data
import callbacks as cb
import logs

num_pairs = 9000
logger = logs.get_tee_logger('test.log')
logger.info('Beginning training module')

logfile = 'runs.log'
logs.log_with_git_hash(' '.join(sys.argv), logfile)

folder = '.'
# Load ibd and accidental data
train_ibd, _, _ = get_ibd_data(tot_num_pairs=num_pairs, just_charges=True,
        train_frac=1, valid_frac=0)
train_acc, _, _ = \
get_ibd_data(path='/project/projectdirs/dasrepo/ibd_pairs/accidentals.h5',
        h5dataset='accidentals_bg_data',
        tot_num_pairs=num_pairs, just_charges=True, train_frac=1, valid_frac=0)
train_set = np.vstack((train_ibd, train_acc))

# Preprocessing: set min, max to -1, 1
min_, max_ = -1, 1
mins, maxes = preprocessing.scale_min_max(train_set, min_, max_)

# Create model
autoencoder, encoder = nn.get_models(16)
nn.compile_model(autoencoder)
models = {'encodings':encoder, 'reconstructions': autoencoder}

# Train
tensorboard = nn.keras.callbacks.TensorBoard(log_dir='batch/logs/tensorboard', write_images=True)
callbacks = [
        tensorboard,
        cb.WriteToLogger(logger),
        cb.SaveIntermediateResults(train_set, models, folder, 5),
        cb.SaveInputs(train_set, folder)
        ]
results = autoencoder.fit(train_set, train_set, epochs=10, batch_size=128,
        callbacks=callbacks)
