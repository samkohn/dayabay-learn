'''The Lasagne NN module for Daya Bay data'''

import numpy as np
import logging
import theano
import theano.tensor as T
import lasagne as l
from operator import mul

from Network import AbstractNetwork
import preprocessing

class IBDPairConvAe(AbstractNetwork):
    '''A convolutional autoencoder for interpreting candidate IBD pairs.'''

    def __init__(self, minibatch_size=128, epochs=1, learn_rate=1e-3,
            bottleneck_width=10, **kwargs):
        '''Initialize a ready-to-train convolutional autoencoder.'''
        super(IBDPairConvAe, self).__init__(self)
        # Shapes are given as (batch, depth, height, width)
        nchannels = kwargs.get('nchannels', 4)
        weighted = kwargs.get('weighted_cost', False)
        self.num_examples = minibatch_size
        self.minibatch_shape = (minibatch_size, nchannels, 8, 24)
        self.minibatch_size = minibatch_size
        self.image_shape = self.minibatch_shape[1:-1]
        self.num_features = reduce(mul, self.image_shape)
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.bottleneck_width = bottleneck_width
        self.input_var = T.dtensor4('input')
        self.network = self._setup_network()
        self.train_prediction = self._setup_prediction(deterministic=False)
        self.test_prediction = self._setup_prediction(deterministic=True)
        self.train_cost = self._setup_cost(deterministic=False,
                weighted=weighted)
        # outputs the cost at each pixel
        self.test_cost_all = self._setup_cost(deterministic=True, array=True)
        self.optimizer = self._setup_optimizer()
        if hasattr(self, 'target_var'):
            inputs = [self.input_var, self.target_var]
        else:
            inputs = [self.input_var]
        self.train_once = theano.function(inputs,
            [self.train_cost], updates=self.optimizer)
        self.predict_fn = theano.function(inputs,
            [self.test_cost_all, self.test_prediction])

    def _setup_network(self):
        '''Construct the ConvAe architecture for Daya Bay IBDs.'''
        network = l.layers.InputLayer(
            input_var=self.input_var,
            name='input',
            shape=self.minibatch_shape)
        network = self._default_network_with_input(network)
        return network

    def _default_network_with_input(self, incoming):
        '''Add on the default/standard conv/bottleneck/deconv layers to the
        specified incoming network'''
        num_filters = 128
        initial_weights = l.init.Normal(1.0/self.num_features, 0)
        network = incoming
        # post-conv shape = (minibatch_size, num_filters, 8, 24)
        network = l.layers.Conv2DLayer(
            network,
            name='conv1',
            num_filters=num_filters,
            filter_size=(5, 5),
            pad=(2, 2),
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # post-pool shape = (minibatch_size, num_filters, 4, 12)
        network = l.layers.MaxPool2DLayer(
            network,
            name='pool1',
            pool_size=(2, 2))
        # post-conv shape = (minibatch_size, num_filters, 4, 10)
        network = l.layers.Conv2DLayer(
            network,
            name='conv2',
            num_filters=num_filters,
            filter_size=(3, 3),
            pad=(1, 0),
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # post-pool shape = (minibatch_size, num_filters, 2, 5)
        network = l.layers.MaxPool2DLayer(
            network,
            name='pool2',
            pool_size=(2, 2))
        # post-conv shape = (minibatch_size, bottleneck_width, 1, 1)
        network = l.layers.Conv2DLayer(
            network,
            name='bottleneck',
            num_filters=self.bottleneck_width,
            filter_size=(2, 5),
            pad=0,
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # post-deconv shape = (minibatch_size, num_filters, 2, 4)
        network = l.layers.Deconv2DLayer(
            network,
            name='deconv1',
            num_filters=num_filters,
            filter_size=(2, 4),
            stride=(2, 2),
            W=initial_weights)
        # post-deconv shape = (minibatch_size, num_filters, 4, 11)
        network = l.layers.Deconv2DLayer(
            network,
            name='deconv2',
            num_filters=num_filters,
            filter_size=(2, 5),
            stride=(2, 2),
            W=initial_weights)
        # post-deconv shape = (minibatch_size, input_depth, 8, 24)
        network = l.layers.Deconv2DLayer(
            network,
            name='deconv3',
            num_filters=self.image_shape[0],
            filter_size=(2, 4),
            stride=(2, 2),
            W=initial_weights)
        return network

    def _setup_prediction(self, deterministic):
        '''Construct the Theano identifier that refers to the output of the
        autoencoder.'''
        return l.layers.get_output(self.network, deterministic=deterministic)

    def _setup_cost(self, deterministic, array=False, weighted=False):
        '''Construct the sum-squared loss between the input and the output.

        Must be called after self.network is defined.'''
        if deterministic:
            prediction = self.test_prediction
        else:
            prediction = self.train_prediction
        cost = l.objectives.squared_error(prediction, self.input_var)
        if weighted:
            # TODO This is super fragile. In particular, we know for IBDConvAe2
            # and its descendents, the data is scaled to be between -1 and 1.
            # So, adding 1 to the input means it will be between 0 and 2, which
            # are good weights. Negative weights seem like a bad idea.
            weights = self.input_var + 1
        else:
            weights = None
        if not array:
            cost = l.objectives.aggregate(cost, weights=weights, mode='mean')
        return cost

    def _setup_optimizer(self):
        '''Construct the gradient descent optimizer.

        Must be called after self.train_cost is defined.'''
        weights = l.layers.get_all_params(self.network, trainable=True)
        updates = l.updates.momentum(
            self.train_cost,
            weights,
            learning_rate=self.learn_rate,
            momentum=0.9)
        return updates


    def minibatch_iterator(self, x, y=None):
        '''Return a generator that goes through the set in mini-batches.

        If y is given, it is interpreted as a vector of target classes, and the
        returned minibatches are tuples of (input, target). If y is not given
        (or is None), then the returned minibatches are simply the ndarray of
        input.
        '''
        if (y is not None) and (x.shape[0] != y.shape[0]):
            raise ValueError("x and y have mismatched shapes: %s vs. %s" %
                    (x.shape, y.shape))
        def minibatches():
            numinputs = x.shape[0]
            indices = np.arange(numinputs)
            # Shuffle order each time a new set of minibatches is requested
            np.random.shuffle(indices)
            # Check for the small-sample case, in which case we don't use a
            # minibatch
            if numinputs > self.minibatch_size:
                upper = numinputs - self.minibatch_size + 1
            else:
                upper = numinputs
            for i in xrange(0, upper, self.minibatch_size):
                excerpt = slice(i, i + self.minibatch_size)
                if y is None:
                    yield x[indices[excerpt]]
                else:
                    yield (x[indices[excerpt]], y[indices[excerpt]])
        return minibatches

    def fit(self, x_train, y_train=None):
        '''Fit and train the autoencoder to the x_train data.

        If y_train is given, assume that it is an array of targets. Obviously
        y_train should not be given if this base class is used, but some
        subclasses use supervised training.'''
        for epoch in xrange(self.epochs):
            minibatches = self.minibatch_iterator(x_train, y_train)
            for batch in minibatches():
                if y_train is None:
                    inputs = batch
                    cost = self.train_once(inputs)[0]
                    predict_fn_args = (x_train[:self.num_examples],)
                else:
                    inputs = batch[0]
                    targets = batch[1]
                    cost = self.train_once(inputs, targets)[0]
                    predict_fn_args = (x_train[:self.num_examples],
                        y_train[:self.num_examples])
            kwargs = {
                'cost': cost,
                'epoch': epoch,
                'input': x_train[:self.num_examples],
                'target': (y_train[:self.num_examples] if y_train is not None
                           else None),
                'output': self.predict_fn(*predict_fn_args)[1]
            }
            for fn in self.epoch_loop_hooks:
                fn(**kwargs)

    def predict(self, x, y=None):
        '''Predict the autoencoded image without training.'''
        # loop by minibatch (this time not randomized)
        output_cost = np.empty(len(x))
        if y is None:
            output_prediction = np.empty_like(x)
        else:
            output_prediction = np.empty((len(y), self.num_classes))

        for i in range(0, len(x)-self.minibatch_size + 1, self.minibatch_size):
            minislice = slice(i, i + self.minibatch_size)
            if y is None:
                prediction = self.predict_fn(x[minislice])
                prediction.append(prediction[0].mean(axis=(1, 2, 3)))
                image_wise_index = len(prediction)-1
            else:
                prediction = self.predict_fn(x[minislice], y[minislice])
            output_cost[minislice] = prediction[image_wise_index]
            output_prediction[minislice] = prediction[1]
        # account for any remaining inputs that don't make a complete minibatch
        num_extras = len(x) % self.minibatch_size
        if num_extras == 0:
            return (output_cost, output_prediction)
        endslice = slice(len(x) - num_extras, len(x))
        fillerslice = slice(0, self.minibatch_size - num_extras)
        x_end = np.vstack((x[endslice], x[fillerslice]))
        if y is None:
            prediction = self.predict_fn(x_end)
            prediction.append(prediction[0].mean(axis=(1, 2, 3)))
            image_wise_index = len(prediction)-1
        else:
            logging.debug('y[endslice].shape = %s', str(y[endslice].shape))
            logging.debug('y[fillerslice].shape = %s', str(y[fillerslice].shape))
            y_end = np.concatenate((y[endslice], y[fillerslice]), axis=0)
            prediction = self.predict_fn(x_end, y_end)
        output_cost[endslice] = prediction[image_wise_index][:num_extras]
        output_prediction[endslice] = prediction[1][:num_extras]
        return (output_cost, output_prediction)

    def extract_layer(self, data, layer):
        '''Extract the output of the given layer.'''
        lasagne_layer = self._get_layer_named(layer)
        output = l.layers.get_output(lasagne_layer)
        out_fn = theano.function([self.input_var], output)
        return out_fn(data)

    def _get_layer_named(self, name):
        '''Get the Lasagne layer object with the given name located in
        self.network.'''

        all_layers = l.layers.get_all_layers(self.network)
        for one_layer in all_layers:
            if one_layer.name == name:
                return one_layer
        raise ValueError('"%s" is not a layer in our network' % name)


    def preprocess_data(self, x, y=None):
        '''Prepare the data for the neural network.

            - Remove 0's from the time channels
            - Center the data on 0
            - Scale it to have a standard deviation of 1'''
        std = 1
        if not getattr(self, 'only_charge', False):
            preprocessing.fix_time_zeros(x)
        means = preprocessing.center(x)
        stds = preprocessing.scale(x, std, mode='standardize')
        def repeat_transformation(other):
            if len(other) == 0:
                return
            else:
                if not getattr(self, 'only_charge', False):
                    preprocessing.fix_time_zeros(other)
                other -= means
                other /= stds/std
        return repeat_transformation

    def save(self, filename):
        params = l.layers.get_all_param_values(self.network)
        np.savez(filename, *params)

    def load(self, filename):
        with np.load(filename) as param_data:
            params = []
            name_prefix = 'arr_'
            for i in xrange(len(param_data.files)):
                name = '%s%d' % (name_prefix, i)
                params.append(param_data[name])
            l.layers.set_all_param_values(self.network, params)


class IBDPairConvAe2(IBDPairConvAe):
    '''A CAE based on IBDPairConvAe that scales input and output to be between
       [-1, 1].'''
    def __init__(self, *args, **kwargs):
        '''Initialize an IBDPairConvAe2.'''
        super(IBDPairConvAe2, self).__init__(*args, **kwargs)

    def _setup_network(self):
        '''Set up the IBDPairConvAe network but have it scale output to +/- 1'''
        network = super(IBDPairConvAe2, self)._setup_network()
        # The "network" is really a pointer to the output deconv layer
        network.nonlinearity = l.nonlinearities.tanh
        return network

    def preprocess_data(self, x, y=None, **kwargs):
        '''Prepare the data for the neural network.

            - Remove 0's from the time channels
            - Center the data on 0
            - Scale it to have lie on the interval [-1, 1]'''
        preprocessing.fix_time_zeros(x)
        means = preprocessing.center(x)
        min_, max_, = -1, 1
        mins, maxes = preprocessing.scale_min_max(x, min_, max_)
        if 'channel' in kwargs:
            channel = kwargs['channel']
        else:
            channel = None
        preprocessing.standardize_cylinder_rotation(x, channel)
        def repeat_transformation(other):
            if len(other) == 0:
                return
            else:
                preprocessing.fix_time_zeros(other)
                other -= means
                other -= mins
                other /= maxes - mins
                other *= max_ - min_
                other += min_
                preprocessing.standardize_cylinder_rotation(other, channel)
        return repeat_transformation

class IBDChargeDenoisingConvAe(IBDPairConvAe2):
    '''A denoising CAE based on IBDPairConvAe2.'''
    def __init__(self, *args, **kwargs):
        '''Initialize the denoising autoencoder.'''
        self.zero_fraction = kwargs.get('zero_fraction', 0.3)
        self.seed = kwargs.get('seed', 498)
        super(IBDChargeDenoisingConvAe, self).__init__(*args, nchannels=2, **kwargs)
        self.only_charge = True

    def _setup_network(self):
        '''Set up the IBDPairConvAe2 network, but have it accept only 2 input
        channels, and partly corrupt the input by zeroing out some of the
        pixels.'''
        # Reassign the first convolutional layer's input to be the new corrupt
        # input (after dropout)
        network = l.layers.InputLayer(
            input_var=self.input_var,
            name='input',
            shape=self.minibatch_shape)
        network = l.layers.DropoutLayer(
                network,
                name='corruptor',
                p=self.zero_fraction)
        network = self._default_network_with_input(network)
        network.nonlinearity = l.nonlinearities.tanh
        return network

class SinglesClassifier(IBDPairConvAe):
    '''A convolutional NN which classifies single triggers in a supervised
    manner.'''
    def __init__(self, *args, **kwargs):
        '''Initialize the network.'''
        nchannels = 1
        self.num_classes = 2
        self.only_charge = True
        self.target_var = T.lvector('target')
        super(SinglesClassifier, self).__init__(*args, nchannels=nchannels, **kwargs)
        #self.train_once = theano.function([self.input_var, self.target_var],
            #[self.train_cost], updates=self.optimizer)

    def _default_network_with_input(self, incoming):
        num_filters = 128
        initial_weights = l.init.Normal(1.0/self.num_features, 0)
        network = incoming
        # post-conv shape = (minibatch_size, num_filters, 8, 24)
        network = l.layers.Conv2DLayer(
            network,
            name='conv1',
            num_filters=num_filters,
            filter_size=(5, 5),
            pad=(2, 2),
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # post-pool shape = (minibatch_size, num_filters, 4, 12)
        network = l.layers.MaxPool2DLayer(
            network,
            name='pool1',
            pool_size=(2, 2))
        # post-conv shape = (minibatch_size, num_filters, 4, 10)
        network = l.layers.Conv2DLayer(
            network,
            name='conv2',
            num_filters=num_filters,
            filter_size=(3, 3),
            pad=(1, 0),
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # post-pool shape = (minibatch_size, num_filters, 2, 5)
        network = l.layers.MaxPool2DLayer(
            network,
            name='pool2',
            pool_size=(2, 2))
        # post-conv shape = (minibatch_size, bottleneck_width, 1, 1)
        network = l.layers.Conv2DLayer(
            network,
            name='bottleneck',
            num_filters=self.bottleneck_width,
            filter_size=(2, 5),
            pad=0,
            W=initial_weights,
            nonlinearity=l.nonlinearities.rectify)
        # One last fully-connected layer
        network = l.layers.DenseLayer(
            network,
            num_units=self.num_classes,
            W=initial_weights,
            nonlinearity=l.nonlinearities.softmax)
        return network
    def _setup_cost(self, deterministic, array=False, **kwargs):
        '''Construct the cross-entropy between the train set and the output.

        Must be called after self.network is defined.'''
        if deterministic:
            prediction = self.test_prediction
        else:
            prediction = self.train_prediction
        cost = l.objectives.categorical_crossentropy(prediction,
                self.target_var)
        if not array:
            cost = l.objectives.aggregate(cost, mode='mean')
        return cost
