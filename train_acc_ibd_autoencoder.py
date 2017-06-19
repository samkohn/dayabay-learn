import numpy as np
import networks.BasicConvAE as nn
import networks.preprocessing as preprocessing
from util.data_loaders import get_ibd_data

num_pairs = 9000
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
tensorboard = nn.keras.callbacks.TensorBoard(log_dir='.', histogram_freq=1,
        write_images=True)
results = autoencoder.fit(train_set, train_set, epochs=4, batch_size=100,
        callbacks=[tensorboard])
