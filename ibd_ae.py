__author__ = 'racah'



# 1) Primary AD           10000 or 1
# 2) Delayed AD response  01000 or 2
# 3) Muon decay           00100 or 3
# 4) Flasher              00010 or 4
# 5) Other (background noise) 00001 or 5



def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10,
        help='number of epochs for training')
    parser.add_argument('-w', '--bottleneck-width', type=int, default=10,
        help='number of features in the bottleneck layer')
    parser.add_argument('-n', '--numpairs', type=int, default=-1,
        help='number of IBD pairs to use')
    parser.add_argument('-p', '--save-prediction', default=None,
        help=('optionally save AE prediction to specified h5' +
        ' file (relative to --out-dir'))
    parser.add_argument('-s', '--save-model', default=None,
        help=('optionally save the trained model parameters to the ' +
        'specified file (relative to --out-dir)'))
    parser.add_argument('-m', '--load-model', default=None,
        help='optionally load a previously saved set of model parameters')
    parser.add_argument('-l', '--learn_rate', default=0.001, type=float,
        help='the learning rate for the network')
    parser.add_argument('--tsne', action='store_true',
        help='do t-SNE visualization')
    parser.add_argument('--cylinder-rotation', default=None,
        help='Argument to pass to preprocessing.standardize_cylinder_rotation')
    parser.add_argument('-v', '--verbose', default=0, action='count',
        help='default:quiet, 1:log_info, 2:+=plots, 3:+=log_debug')
    parser.add_argument('--logfile', default='./runs.log',
        help='location of the file to log each run (NOT all logger output)')
    parser.add_argument('--out-dir', default='.',
        help='directory to save all files that may be requested')
    parser.add_argument('--save-interval', default=10, type=int,
        help='number of epochs between saving intermediate outputs')
    parser.add_argument('--weighted-cost', action='store_true',
        help='weight the costs by the individual pixel weights')
    parser.add_argument('--network', default='IBDPairConvAe',
        choices=[
            'IBDPairConvAe',
            'IBDPairConvAe2',
            'IBDChargeDenoisingConvAe',
            'SinglesClassifier',
        ],
        help='network to use')
    parser.add_argument('--accidental-fraction', type=float, default=0,
        help='fraction of train, test, and val sets that are' +
        ' intentionally accidentals')
    parser.add_argument('--accidental-location', default=None,
        help='file path of accidentals h5 file')
    parser.add_argument('--train-val-test', nargs=3, type=float,
            default=[0.5, 0.25, 0.25],
            metavar=('TRAIN_FRAC', 'VAL_FRAC', 'TEST_FRAC'),
            help='how to divide the data set between training, evaluation, ' +
            'and test data sets')
    return parser

if __name__ == "__main__":
    import argparse
    import logging
    import ast
    logging.basicConfig(format='%(levelname)s:\t%(message)s')
    parser = setup_parser()
    args = parser.parse_args()
    # enforce that the train-val-test arguments must sum to 1
    if round(sum(args.train_val_test), 5) != 1.0:
        raise ValueError('--train-val-test fractions must sum to 1')
    if args.cylinder_rotation is None:
        pass
    else:
        args.cylinder_rotation = ast.literal_eval(args.cylinder_rotation)
    import numpy as np
    import os
    import pickle
    import sys
    import subprocess
    import h5py
    import matplotlib
    from sklearn.manifold import TSNE
    import numpy as np
    matplotlib.use('agg')
    from matplotlib import pyplot as plt
    from sklearn.decomposition import PCA
    from vis.viz import Viz
    from util.data_loaders import load_ibd_pairs, get_ibd_data
    from util.helper_fxns import make_accidentals
    from networks.LasagneConv import IBDPairConvAe, IBDPairConvAe2
    from networks.LasagneConv import IBDChargeDenoisingConvAe
    from networks.LasagneConv import SinglesClassifier

    make_progress_plots = False
    if args.verbose == 0:
        pass
    elif args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)
    elif args.verbose == 2:
        logging.getLogger().setLevel(logging.INFO)
        make_progress_plots = True
    else:
        logging.getLogger().setLevel(logging.DEBUG)
        make_progress_plots = True

    # Save the specific command run to the log file with a date and
    # time stamp, and the git commit hash used.
    commit_hash = subprocess.check_output(['git', 'describe',
        '--always']).strip().split('-')[-1]
    runlogger = logging.getLogger('runlogger')
    runlogger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)-35s %(message)s',
            datefmt='%Y-%b-%d %H:%M, git ' + commit_hash + ']')
    handler = logging.FileHandler(args.logfile)
    handler.setFormatter(formatter)
    runlogger.addHandler(handler)
    runlogger.debug(' '.join(sys.argv))
    supervised = set(['SinglesClassifier'])
    train_frac, val_frac, test_frac = args.train_val_test

    #class for networks architecture
    logging.info('Constructing untrained ConvNet of class %s', args.network)
    convnet_class = eval(args.network)
    cae = convnet_class(bottleneck_width=args.bottleneck_width,
        epochs=args.epochs, learn_rate=args.learn_rate,
        weighted_cost=args.weighted_cost)
    if args.load_model:
        logging.info('Loading model parameters from file %s', args.load_model)
        cae.load(args.load_model)
    logging.info('Preprocessing data files')
    only_charge = getattr(cae, 'only_charge', False)
    num_ibds = int(round((1 - args.accidental_fraction) * args.numpairs))
    train, val, test = get_ibd_data(tot_num_pairs=num_ibds,
        just_charges=only_charge, train_frac=train_frac, valid_frac=val_frac)
    train_IBD = train
    val_IBD = val
    test_IBD = test
    if args.network == 'SinglesClassifier':
        train = train[:, 1:2, :, :]
        val = val[:, 1:2, :, :]
        test = test[:, 1:2, :, :]
    if args.accidental_fraction > 0:
        num_accidentals = args.numpairs - num_ibds
        if args.accidental_location is None:
            path='/global/homes/s/skohn/ml/dayabay-data-conversion/extract_accidentals/accidentals3.h5'
        else:
            path = args.accidental_location
        dsetname='accidentals_bg_data'
        train_acc, val_acc, test_acc = get_ibd_data(
                path=path, tot_num_pairs=num_accidentals,
                just_charges=only_charge, h5dataset=dsetname,
                train_frac=train_frac, valid_frac=val_frac)
        if args.network == 'SinglesClassifier':
            train_acc = train_acc[:, 0:1, :, :]
            val_acc = val_acc[:, 0:1, :, :]
            test_acc = test_acc[:, 0:1, :, :]
        train = np.vstack((train, train_acc))
        val = np.vstack((val, val_acc))
        test = np.vstack((test, test_acc))
    preprocess = cae.preprocess_data(train, channel=args.cylinder_rotation)
    preprocess(val)
    preprocess(test)

    if args.network == 'SinglesClassifier':
        train_targets = np.zeros((train.shape[0]), dtype=int)
        val_targets = np.zeros((val.shape[0]), dtype=int)
        test_targets = np.zeros((test.shape[0]), dtype=int)
        train_targets[train_IBD.shape[0]:] = 1
        val_targets[val_IBD.shape[0]:] = 1
        test_targets[test_IBD.shape[0]:] = 1
        #train = train[:, 1:2, :, :]
        #val = val[:, 1:2, :, :]
        #test = test[:, 1:2, :, :]


    # set up a decorator to only run the function if the epoch is at the
    # appropriate value (usually == 0 (mod 10) or some such thing)
    def only_on_some_epochs(f):
        def wrapper(*arglist, **kwargs):
            if kwargs['epoch'] % args.save_interval == args.save_interval - 1:
                return f(*arglist, **kwargs)
            else:
                return
        return wrapper

    #uses scikit-learn interface (so this trains on X_train)
    epochs = []
    costs = []
    def record_cost_curve(**kwargs):
        epochs.append(kwargs['epoch'])
        costs.append(kwargs['cost'])

    @only_on_some_epochs
    def plot_cost_curve(**kwargs):
        plt.plot(epochs, costs)
        plt.savefig(os.path.join(args.out_dir, 'cost.pdf'))
        plt.clf()

    @only_on_some_epochs
    def saveparameters(**kwargs):
        cae.save(os.path.join(args.out_dir, args.save_model +
            str(kwargs['epoch'])))

    @only_on_some_epochs
    def plotcomparisons(**kwargs):
        numevents = 6
        plotargs = {
            'interpolation': 'nearest',
            'aspect': 'auto',
            'vmin': -1,
            'vmax': 1,
        }
        if kwargs['input'].shape[1] == 4:
            delayed_index = 2
        else:
            delayed_index = 1
        for i in range(numevents):
            fig = plt.figure(1)
            plt.subplot(4, numevents, i + 1)
            plt.imshow(kwargs['input'][i, 0].T, **plotargs)
            plt.title('input %d' % i)
            if i == 0:
                plt.ylabel('Prompt Charges')
            plt.subplot(4, numevents, numevents + i + 1)
            plt.imshow(kwargs['input'][i, delayed_index].T, **plotargs)
            if i == 0:
                plt.ylabel('Delayed Charges')
            plt.subplot(4, numevents, i + 2 * numevents + 1)
            plt.imshow(kwargs['output'][i, 0].T, **plotargs)
            plt.title('output %d' % i)
            if i == 0:
                plt.ylabel('Prompt Charges')
            plt.subplot(4, numevents, i + 3 * numevents + 1)
            plt.imshow(kwargs['output'][i, delayed_index].T, **plotargs)
            if i == 0:
                plt.ylabel('Delayed Charges')
        fig.set_size_inches(12, 16, forward=True)
        plt.savefig(os.path.join(
            args.out_dir,
            'reco%d.pdf' % kwargs['epoch']))
        plt.clf()

    def log_message_cost(**kwargs):
        logging.debug('Loss after epoch %d is %f', kwargs['epoch'],
            kwargs['cost'])
    cae.epoch_loop_hooks.append(log_message_cost)
    if make_progress_plots:
        cae.epoch_loop_hooks.append(record_cost_curve)
        cae.epoch_loop_hooks.append(plot_cost_curve)
        if args.network not in supervised:
            cae.epoch_loop_hooks.append(plotcomparisons)
    if args.save_model:
        cae.epoch_loop_hooks.append(saveparameters)
    logging.info('Training network with %d samples', train.shape[0])
    if args.network in supervised:
        cae.fit(train, train_targets)
    else:
        cae.fit(train)


    if args.tsne:
        logging.info('Constructing visualization')
        v = Viz(gr_truth,nclass=1)

        # take first two principal components of features, so we can plot easily
        #normally we would do t-SNE (but takes too long for quick demo)
        #x_pc = v.get_pca(feat)

        num_feats = 500 if feat.shape[0] > 500 else feat.shape[0]
        x_ts = v.get_tsne(feat[:num_feats])

        #plot the 2D-projection of the features
        v.plot_features(x_ts,save=True)

    if args.save_prediction is not None:
        logging.info('Saving autoencoder output')
        if args.network in supervised:
            logging.debug('train.shape = %s', str(train.shape))
            logging.debug('train targets.shape = %s', str(train_targets.shape))
            train_cost_prediction = cae.predict(train, train_targets)
            logging.debug('val.shape = %s', str(val.shape))
            logging.debug('val targets.shape = %s', str(val_targets.shape))
            val_cost_prediction = cae.predict(val, val_targets)
            logging.debug('test.shape = %s', str(test.shape))
            logging.debug('test targets.shape = %s', str(test_targets.shape))
            test_cost_prediction = cae.predict(test, test_targets)
        else:
            train_cost_prediction = cae.predict(train)
            val_cost_prediction = cae.predict(val)
            test_cost_prediction = cae.predict(test)
        outdata = np.vstack((train_cost_prediction[1], val_cost_prediction[1],
            test_cost_prediction[1]))
        outcosts = np.concatenate((train_cost_prediction[0], val_cost_prediction[0],
            test_cost_prediction[0]), axis=0)
        indata = np.vstack((train, val, test))
        filename = os.path.join(args.out_dir, args.save_prediction)
        outfile = h5py.File(filename, 'w')
        indset = outfile.create_dataset("ibd_pair_inputs", data=indata,
            compression="gzip", chunks=True)
        outdset = outfile.create_dataset("ibd_pair_predictions", data=outdata,
            compression="gzip", chunks=True)
        costdset = outfile.create_dataset("costs", data=outcosts,
            compression="gzip", chunks=True)
        outfile.close()
