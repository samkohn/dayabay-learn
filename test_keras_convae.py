import numpy as np
import networks.BasicConvAE as nn
import networks.preprocessing as preprocessing
from util.data_loaders import get_ibd_data

# Load ibd and accidental data
train_ibd, _, _ = get_ibd_data(tot_num_pairs=2000, just_charges=True,
        train_frac=1, valid_frac=0)
train_acc, _, _ = \
get_ibd_data(path='/project/projectdirs/dasrepo/ibd_pairs/accidentals.h5',
        h5dataset='accidentals_bg_data',
        tot_num_pairs=2000, just_charges=True, train_frac=1, valid_frac=0)
train_set = np.vstack((train_ibd, train_acc))
#train_set = train_ibd

# Preprocessing: set mean to 0 and min, max to -1, 1
means = preprocessing.center(train_set)
min_, max_ = -1, 1
mins, maxes = preprocessing.scale_min_max(train_set, min_, max_)

# Create model
model = nn.get_model(256)
nn.compile_model(model)

# Train
def printweights(epoch, logs):
    print ""
    print model.get_weights()[0][1][0][0][5]
    return
debug_callback = nn.keras.callbacks.LambdaCallback(
        on_epoch_begin=printweights)
tensorboard = nn.keras.callbacks.TensorBoard(log_dir='.', histogram_freq=1,
        write_images=True)
results = model.fit(train_set, train_set, epochs=10, batch_size=100,
        callbacks=[debug_callback, tensorboard])
