import numpy as np
import h5py
import networks.BasicConvAE as nn
import networks.preprocessing as preprocessing
from util.data_loaders import get_ibd_data
import logs

num_pairs = 9000
logger = logs.get_tee_logger('test.log')
logger.info('Beginning training module')

logfile = 'runs.log'
logs.log_with_git_hash(' '.join(sys.argv), logfile)

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

# Train
tensorboard = nn.keras.callbacks.TensorBoard(log_dir='batch/logs/tensorboard', write_images=True)
results = autoencoder.fit(train_set, train_set, epochs=500, batch_size=128,
        callbacks=[tensorboard])
num_to_save = 1000
encodings = encoder.predict(train_set[:num_to_save])
reconstructions = autoencoder.predict(train_set[:num_to_save])
autoencoder.save_weights('output_weights.h5')
with h5py.File('output.h5') as outfile:
    input_dataset = outfile.create_dataset('input',
            data=train_set[:num_to_save],
            compression='gzip', chunks=True)
    encodings_dataset = outfile.create_dataset('encodings', data=encodings,
            compression='gzip', chunks=True)
    reconstructions_dataset = outfile.create_dataset('reconstructions',
            data=reconstructions, compression='gzip', chunks=True)
