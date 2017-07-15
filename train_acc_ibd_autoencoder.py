# First parse all the arguments to quit fast if there's an error
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=10,
        help='number of epochs to train for')
parser.add_argument('-n', '--numpairs', type=int, default=1000,
        help='number of IBD/event pairs total to use')
parser.add_argument('-w', '--bottleneck-width', type=int, default=16,
        help='number of features in the bottleneck layer')
parser.add_argument('--master-logfile', default='master.log',
        help='logfile to keep track of runs')
parser.add_argument('--run-logfile', default='test.log',
        help='logfile to monitor this run')
parser.add_argument('-o', '--output', default='.',
        help='folder to save all files generated in this run')
parser.add_argument('--save-interval', type=int, default=10,
        help='number of epochs between saving intermediate output')
parser.add_argument('--cylinder-rotation', default=None,
        help='Argument to pass to preprocessing.standardize_cylinder_rotation')
args = parser.parse_args()

# Import all other modules
import numpy as np
import h5py
import sys
import os
import networks.BasicConvAE as nn
import networks.BasicConvAE2 as nn2
import networks.preprocessing as preprocessing
from util.data_loaders import get_ibd_data
import callbacks as cb
import logs
import ast # used to parse --cylinder-rotation arg

output_folder = args.output
logger = logs.get_tee_logger(os.path.join(output_folder, args.run_logfile))
logs.log_with_git_hash(' '.join(sys.argv), args.master_logfile)
numpairs = args.numpairs
epochs = args.epochs
save_interval = args.save_interval
# Parse the "cylinder-rotation" argument into a dict
if args.cylinder_rotation is not None:
    args.cylinder_rotation = ast.literal_eval(args.cylinder_rotation)

logger.info('Beginning training module')

# Load ibd and accidental data
num_ibds = numpairs/2
num_acc = numpairs/2
train_ibd, _, _ = get_ibd_data(tot_num_pairs=num_ibds, just_charges=True,
        train_frac=1, valid_frac=0)
train_acc, _, _ = \
get_ibd_data(path='/project/projectdirs/dasrepo/ibd_pairs/accidentals.h5',
        h5dataset='accidentals_bg_data',
        tot_num_pairs=num_acc, just_charges=True, train_frac=1, valid_frac=0)
train_set = np.vstack((train_ibd, train_acc))
classes = np.hstack((np.zeros(num_ibds), np.ones(num_acc)))

# Preprocessing: set min, max to -1, 1
min_, max_ = -1, 1
mins, maxes = preprocessing.scale_min_max(train_set, min_, max_)

# Preprocessing: cyclically permute ("rotate") images
train_set = preprocessing.standardize_cylinder_rotation(train_set,
        channel=args.cylinder_rotation)

# Create model
autoencoder, encoder = nn2.get_models(16)
nn2.compile_model(autoencoder)
models = {'encodings':encoder, 'reconstructions': autoencoder}

# Train
tensorboard = nn.keras.callbacks.TensorBoard(log_dir='batch/logs/tensorboard', write_images=True)
callbacks = [
        tensorboard,
        cb.WriteToLogger(logger),
        cb.SaveIntermediateResults(train_set, models, output_folder,
            save_interval),
        cb.SaveInputs(train_set, classes, output_folder),
        cb.SaveCostCurve(output_folder, save_interval)
        ]
results = autoencoder.fit(train_set, train_set, epochs=epochs, batch_size=128,
        callbacks=callbacks)
