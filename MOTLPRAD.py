import time
import timeit
import numpy
import lasagne
import theano

import theano.tensor as T
from lasagne.nonlinearities import rectify
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score

import tensorflow as tf
devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

import pandas as pd
import numpy as np
import keras
from keras import Input, Model
from keras.layers import Dropout, Dense, Activation, Lambda
from tensorflow.keras.optimizers import SGD
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
import pickle
import Initialization

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """
    def __init__(self, n_in,
             learning_rate, hidden_layers_sizes=None,
             lr_decay=0.0, momentum=0.9,
             L2_reg=0.0, L1_reg=0.0,
             activation="rectify",
             dropout=None,
             batch_norm=False,
             standardize=False,
             numpy_rng=None,
             theano_rng=None):
        self.X = T.fmatrix('X')  # patients covariates
        self.y = T.ivector('Y')  # the observations vector
        self.is_train = T.iscalar('is_train')

        ## sdA_hidden_layers, a HiddenLayer object is used to store the layers shared by stacked auto-encoders
        self.hidden_layers = []
        ## where to store the auto-encoder
        self.dA_layers = []
        self.n_layers = len(hidden_layers_sizes)
        self.hidden_layers_sizes = hidden_layers_sizes
        self.L1 = 0
        self.L2 = 0

        if numpy_rng is None:
            numpy_rng = numpy.random.RandomState(11111)
        activation_fn = rectify
        self.params = []

        for idx, hidden_layer_size in enumerate(hidden_layers_sizes):
            if idx == 0:
                input_size = n_in
                layer_input = self.X
            else:
                input_size = hidden_layers_sizes[idx - 1]
                layer_input = self.hidden_layers[-1].output

            if dropout and dropout > 0:
                hidden_layer = DropoutHiddenLayer(rng=numpy_rng,
                                                  input=layer_input,
                                                  n_in=input_size,
                                                  n_out=hidden_layers_sizes[idx],
                                                  activation=activation_fn,
                                                  dropout_rate=dropout,
                                                  is_train=self.is_train)
            else:
                hidden_layer = HiddenLayer(rng=numpy_rng,
                                           input=layer_input,
                                           n_in=input_size,
                                           n_out=hidden_layers_sizes[idx],
                                           activation=activation_fn)

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[idx],
                          W=hidden_layer.W,
                          bhid=hidden_layer.b,
                          non_lin=activation_fn)
            self.dA_layers.append(dA_layer)

            self.L1 += abs(hidden_layer.W).sum()
            self.L2 += (hidden_layer.W ** 2).sum()

        # Adds a risk prediction layer on top of the stack.
        if self.n_layers == 0:
            self.logRegressionLayer = LogisticRegression(
                input=self.X,
                n_in=n_in,
                n_out=2
            )
        else:
            self.logRegressionLayer = LogisticRegression(
                input=self.hidden_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=2)
        self.L1 += abs(self.logRegressionLayer.W).sum()
        self.L2 += (self.logRegressionLayer.W ** 2).sum()
        self.params.extend(self.logRegressionLayer.params)

        self.regularizers = L1_reg * self.L1 + L1_reg * self.L2
        self.n_in = n_in
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.L2_reg = L2_reg
        self.L1_reg = L1_reg
        self.momentum = momentum
        self.dropout = dropout

        self.finetune_cost = self.logRegressionLayer.negative_log_likelihood(self.y)
        self.errors = self.logRegressionLayer.errors
        self.hyperparams = {
            'n_in': n_in,
            'learning_rate': learning_rate,
            'hidden_layers_sizes': hidden_layers_sizes,
            'lr_decay': lr_decay,
            'momentum': momentum,
            'L2_reg': L2_reg,
            'L1_reg': L1_reg,
            'activation': activation,
            'dropout': dropout,
            'batch_norm': batch_norm,
            'standardize': standardize
        }

    def pretraining_functions(self, pretrain_x, batch_size):
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption
        learning_rate = T.scalar('lr')  # learning rate

        pretrain_x_batch = pretrain_x
        if batch_size:
            # begining of a batch, given `index`
            batch_begin = index * batch_size
            # ending of a batch given `index`
            batch_end = batch_begin + batch_size
            pretrain_x_batch = pretrain_x[batch_begin: batch_end]

        pretrain_fns = []
        is_train = numpy.cast['int32'](0)   # value does not matter
        for dA_layer in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA_layer.get_cost_updates(corruption_level,
                                                      learning_rate)
            # compile the theano function
            fn = theano.function(
                on_unused_input='ignore',
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.X: pretrain_x_batch,
                    self.is_train: is_train
                }
            )
            pretrain_fns.append(fn)

        return pretrain_fns

    def pretrain(self, pretrain_set, pretrain_config=None, verbose=False):
        n_layers = len(self.dA_layers)
        if pretrain_config is not None:
            n_batches = pretrain_set.get_value(borrow=True).shape[0]
            n_batches //= pretrain_config['pt_batchsize']

            pretraining_fns = self.pretraining_functions(
                pretrain_set,
                pretrain_config['pt_batchsize'])
            start_time = timeit.default_timer()
            # de-noising level
            corruption_levels = [pretrain_config['corruption_level']] * n_layers
            for i in range(n_layers):  # Layerwise pre-training
                # go through pretraining epochs
                for epoch in range(pretrain_config['pt_epochs']):
                    # go through the training set
                    c = []
                    for batch_index in range(n_batches):
                        c.append(pretraining_fns[i](index=batch_index,
                                                    corruption=corruption_levels[i],
                                                    lr=pretrain_config['pt_lr']))

                    if verbose:
                        print ('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c, dtype='float64')))

            end_time = timeit.default_timer()
            if verbose:
                print('Pretraining took {} minutes.'.format((end_time - start_time) / 60.))

    def build_finetune_functions(self, learning_rate, update_fn, istune=False):

        is_train = T.iscalar('is_train')
        X = T.matrix('X', dtype='float32')
        y = T.ivector('Y')

        loss = self.finetune_cost + self.regularizers
        if istune:
            loss = self.finetune_cost
        updates = update_fn(loss, self.params, learning_rate=learning_rate)
        if self.momentum:
            updates = lasagne.updates.apply_nesterov_momentum(updates,
                self.params, momentum=self.momentum)

        test = theano.function(
            on_unused_input='ignore',
            inputs=[X, y, is_train],
            outputs=[self.finetune_cost, self.errors(self.y),
                     self.logRegressionLayer.output, self.logRegressionLayer.input],
            givens={
                self.X: X,
                self.y: y,
                self.is_train: is_train
            },
            name='test'
        )

        train = theano.function(
            on_unused_input='ignore',
            inputs=[X, y, is_train],
            outputs=[self.finetune_cost, self.errors(y),
                     self.logRegressionLayer.output, self.logRegressionLayer.input],
            updates=updates,
            givens={
                self.X: X,
                self.y: y,
                self.is_train: is_train
            },
            name='train'
        )


        return train, test

    def get_params(self):
        return [param.copy().eval() for param in self.params]

    def reset_weight(self, params):
        for i in range(self.n_layers):
            self.hidden_layers[i].reset_weight((params[2 * i], params[2 * i + 1]))
        self.logRegressionLayer.reset_weight(params[-2:])

    def train(self,
              train_data, valid_data=None,
              n_epochs=500,
              validation_frequency=250,
              patience=2000, improvement_threshold=0.99999, patience_increase=2,
              batch_size=20,
              update_fn=lasagne.updates.sgd,
              **kwargs):

        x_train, y_train = train_data
        n_train_batches = x_train.shape[0]
        n_train_batches //= batch_size

        n_val_batches = n_train_batches
        if valid_data:
            x_valid, y_valid = valid_data
            n_val_batches = x_valid.shape[0]
            n_val_batches //= batch_size

        best_validation_loss = numpy.inf
        best_params = None

        # Initialize Training Parameters
        lr = theano.shared(numpy.asarray(self.learning_rate,
                                    dtype = numpy.float64))
        # momentum = numpy.array(0, dtype= numpy.float32)

        train_fn, valid_fn = self.build_finetune_functions(
            learning_rate=lr,
            update_fn=update_fn
        )

        def train_batch(index):
            x_train_batch = x_train[index * batch_size:(index + 1) * batch_size]
            y_train_batch = y_train[index * batch_size:(index + 1) * batch_size]
            cost, err, output, input = train_fn(x_train_batch, y_train_batch, 1)
            return err

        def validate_model():
            res = []
            for index in range(n_val_batches):
                x_train_batch = x_valid[index * batch_size:(index + 1) * batch_size]
                y_train_batch = y_valid[index * batch_size:(index + 1) * batch_size]
                cost, errs, output, input = train_fn(x_train_batch, y_train_batch, 0)
                res.append(errs)
            return res

        start = time.time()
        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_batch(minibatch_index)

                iter = (epoch - 1) * n_train_batches + minibatch_index
                if valid_data and (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    this_validation_loss = numpy.mean(validation_losses, dtype='float64')
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                        ):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_params = [param.copy().eval() for param in self.params]
                        best_iter = iter

                if patience <= iter:
                    done_looping = True
                    break

            decay_learning_rate = theano.function(
                inputs=[], outputs=lr,
                updates={lr: lr * (1 / (1 + self.lr_decay))})
            decay_learning_rate()

        if valid_data and best_params:
            for idx, param in enumerate(self.params):
                param.set_value(best_params[idx])

    def tune(self,
              train_data, valid_data=None,
              n_epochs=500,
              validation_frequency=250,
              batch_size=32,
              update_fn=lasagne.updates.sgd,
              **kwargs):

        x_train, y_train = train_data

        batch_size_new = min(x_train.shape[0], batch_size)
        n_train_batches = x_train.shape[0]
        n_train_batches //= batch_size_new
        best_params = None

        # Initialize Training Parameters
        lr = theano.shared(numpy.asarray(self.learning_rate,
                                    dtype = numpy.float64))
        # momentum = numpy.array(0, dtype= numpy.float32)

        train_fn, valid_fn = self.build_finetune_functions(
            learning_rate=lr,
            update_fn=update_fn,
            istune=True
        )

        def train_batch(index):
            x_train_batch = x_train[index * batch_size:(index + 1) * batch_size]
            y_train_batch = y_train[index * batch_size:(index + 1) * batch_size]
            cost, err, output, input = train_fn(x_train_batch, y_train_batch, 1)
            return err

        for epoch in range(n_epochs):
            for minibatch_index in range(n_train_batches):
                minibatch_avg_cost = train_batch(minibatch_index)

            decay_learning_rate = theano.function(
                inputs=[], outputs=lr,
                updates={lr: lr * (1 / (1 + self.lr_decay))})
            decay_learning_rate()


    def get_score(self, X, is_train=0):

        score = theano.function(
            on_unused_input='ignore',
            inputs=[self.X, self.is_train],
            outputs=self.logRegressionLayer.output,
            name='score'
        )

        return score(X, is_train)

    def get_pred(self, X, is_train=0):

        score = theano.function(
            on_unused_input='ignore',
            inputs=[self.X, self.is_train],
            outputs=self.logRegressionLayer.y_pred,
            name='pred'
        )

        return score(X, is_train)
        
    def get_auc(self, test_data, race=None):
        x_test, y_test, r_test = test_data
        y_scr = self.get_score(x_test)[:,1]
        y_ture = y_test
        if race:
            idx = r_test == race
            y_scr = y_scr[idx]
            y_ture = y_test[idx]

        return roc_auc_score(list(y_ture), list(y_scr))


def get_k_best(X_train, y_train, X_test, k=400):
    k_best = SelectKBest(f_classif, k=k)
    k_best.fit(X_train, y_train)
    res = (k_best.transform(X_train),
           k_best.transform(X_test))
    return res

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        Hidden unit activation is given by: activation(dot(input,W) + b)
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :type n_in: int
        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other functions, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]

    def reset_weight(self, params):
        self.W.set_value(params[0])
        self.b.set_value(params[1])

    def reset_weight_by_rate(self, rate):
        if rate != 0:
            self.W.set_value(self.W.get_value() / rate)
            self.b.set_value(self.b.get_value() / rate)

from theano.ifelse import ifelse
import numpy as np

class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, is_train,
                 activation, dropout_rate, mask=None, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.dropout_rate = dropout_rate
        self.srng = T.shared_randomstreams.RandomStreams(rng.randint(999999))
        self.mask = mask
        self.layer_output = self.output

        # Computes outputs for train and test phase applying dropout when needed.
        # train_output = self.layer_output * T.cast(self.mask, theano.config.floatX)
        train_output = self.drop(self.layer_output, self.dropout_rate)
        test_output = self.output * (1 - dropout_rate)
        self.output = ifelse(T.eq(is_train, 1), train_output, test_output)
        return

    def drop(self, input, p=0.5):
        """
        :type input: numpy.array
        :param input: layer or weight matrix on which dropout resp. dropconnect is applied

        :type p: float or double between 0. and 1.
        :param p: p probability of NOT dropping out a unit or connection, therefore (1.-p) is the drop rate.
        """

        mask = self.srng.binomial(n=1, p=p, size=input.shape, dtype=theano.config.floatX)
        return input * mask

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

        # the output of the softmax layer.
        self.output = T.nnet.softmax(T.dot(input, self.W) + self.b)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def reset_weight(self, params):
        self.W.set_value(params[0])
        self.b.set_value(params[1])

    def reset_weight_by_rate(self, rate):
        if rate != 0:
            self.W.set_value(self.W.get_value() / rate)
            self.b.set_value(self.b.get_value() / rate)

from theano.tensor.shared_randomstreams import RandomStreams


class dA(object):
    """Sparse Denoising Auto-Encoder class (dA)
    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Refer to Vincent et al.,2008 for
    details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.
    .. math::
        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)
        y = s(W \tilde{x} + b)                                           (2)
        x = s(W' y  + b')                                                (3)
        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,
        non_lin=None,
        ce=False
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer.
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`
        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA
        :type n_visible: int
        :param n_visible: number of visible units
        :type n_hidden: int
        :param n_hidden:  number of hidden units
        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None
        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        :type ce: boolean
        :param ce: Boolean determining whether to use cross entropy or
                    mean squared error for cost

        """
        self.non_lin = non_lin
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.ce = ce
        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

            # note : W' was written as `W_prime` and b' as `b_prime`
            if not W:
                # W is initialized with `initial_W` which is uniformely sampled
                # from -4*sqrt(6./(n_visible+n_hidden)) and
                # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
                # converted using asarray to dtype
                # theano.config.floatX so that the code is runable on GPU
                initial_W = numpy.asarray(
                    numpy_rng.uniform(
                        low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                        high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                        size=(n_visible, n_hidden)
                    ),
                    dtype=theano.config.floatX
                )
                W = theano.shared(value=initial_W, name='W', borrow=True)

            if not bvis:
                bvis = theano.shared(
                    value=numpy.zeros(
                        n_visible,
                        dtype=theano.config.floatX
                    ),
                    borrow=True
                )

            if not bhid:
                bhid = theano.shared(
                    value=numpy.zeros(
                        n_hidden,
                        dtype=theano.config.floatX
                    ),
                    name='b',
                    borrow=True
                )

            self.W = W
            # b corresponds to the bias of the hidden
            self.b = bhid
            # b_prime corresponds to the bias of the visible
            self.b_prime = bvis
            # tied weights, therefore W_prime is W transpose
            self.W_prime = self.W.T
            self.theano_rng = theano_rng
            # if no input is given, generate a variable representing the input
            if input is None:
                # we use a matrix because we expect a minibatch of several
                # examples, each example being a row
                self.x = T.dmatrix(name='input')
            else:
                self.x = input

            self.params = [self.W, self.b, self.b_prime]

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps ``1 - corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial
                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``
                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.
        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input
    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return self.non_lin((T.dot(input, self.W) + self.b))

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return self.non_lin((T.dot(hidden, self.W_prime) + self.b_prime))

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        if (self.ce):
            L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        else:
            L = T.sum((self.x - z) ** 2, axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy or mean squared error cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)

from scipy.io import savemat, loadmat
def tumor_types(cancer_type):
    Map = {'GBMLGG': ['GBM', 'LGG'],
           'COADREAD': ['COAD', 'READ'],
           'KIPAN': ['KIRC', 'KICH', 'KIRP'],
           'STES': ['ESCA', 'STAD'],
           'PanGI': ['COAD', 'STAD', 'READ', 'ESCA'],
           'PanGyn': ['OV', 'CESC', 'USC', 'UCEC'],
           'PanSCCs': ['LUSC', 'HNSC', 'ESCA', 'CESC', 'BLCA'],
           'PanPan': ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC',
                           'ESCA', 'GBM', 'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG',
                           'LIHC', 'LUAD', 'LUSC', 'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ',
                           'SARC', 'SKCM', 'STAD', 'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM']
           }
    if cancer_type not in Map:
        Map[cancer_type] = [cancer_type]

    return Map[cancer_type]

def get_race(cancer_type):
    path = 'Genetic_Ancestry.xlsx'
    df_list = [pd.read_excel(path, disease, usecols='A,E', index_col='Patient_ID', keep_default_na=False)
               for disease in tumor_types(cancer_type)]
    df_race = pd.concat(df_list)
    df_race = df_race[df_race['EIGENSTRAT'].isin(['EA', 'AA', 'EAA', 'NA', 'OA'])]
    df_race['race'] = df_race['EIGENSTRAT']

    df_race.loc[df_race['EIGENSTRAT'] == 'EA', 'race'] = 'WHITE'
    df_race.loc[df_race['EIGENSTRAT'] == 'AA', 'race'] = 'BLACK'
    df_race.loc[df_race['EIGENSTRAT'] == 'EAA', 'race'] = 'ASIAN'
    df_race.loc[df_race['EIGENSTRAT'] == 'NA', 'race'] = 'NAT_A'
    df_race.loc[df_race['EIGENSTRAT'] == 'OA', 'race'] = 'OTHER'
    df_race = df_race.drop(columns=['EIGENSTRAT'])

    return df_race

def add_race_CT(cancer_type, df, target, groups):
    df_race = get_race(cancer_type)
    df_race = df_race[df_race['race'].isin(groups)]
    df_C_T = get_CT(target)

    # Keep patients with race information
    df = df.join(df_race, how='inner')
    print(df.shape)
    df = df.dropna(axis='columns')
    df = df.join(df_C_T, how='inner')
    print(df.shape)

    # Packing the data
    C = df['C'].tolist()
    R = df['race'].tolist()
    T = df['T'].tolist()
    E = [1 - c for c in C]
    df = df.drop(columns=['C', 'race', 'T'])
    X = df.values
    X = X.astype('float32')
    data = {'X': X, 'T': np.asarray(T, dtype=np.float32),
            'C': np.asarray(C, dtype=np.int32), 'E': np.asarray(E, dtype=np.int32),
            'R': np.asarray(R), 'Samples': df.index.values, 'FeatureName': list(df)}

    return data
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

def get_one_race(dataset, race):
    X, T, C, E, R = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R']
    mask = R == race
    X, T, C, E, R = X[mask], T[mask], C[mask], E[mask], R[mask]
    data = {'X': X, 'T': T, 'C': C, 'E': E, 'R': R}
    return data

def get_CT(target):
    path1 = 'TCGA-CDR-SupplementalTableS1.xlsx'
    cols = 'B,Z,AA'
    if target == 'DSS':
        cols = 'B,AB,AC'
    elif target == 'DFI':
        cols = 'B,AD,AE'
    elif target == 'PFI':
        cols = 'B,AF,AG'

    df_C_T = pd.read_excel(path1, 'TCGA-CDR', usecols=cols, index_col='bcr_patient_barcode')
    df_C_T.columns = ['E', 'T']
    df_C_T = df_C_T[df_C_T['E'].isin([0, 1])]
    df_C_T = df_C_T.dropna()
    df_C_T['C'] = 1 - df_C_T['E']
    df_C_T.drop(columns=['E'], inplace=True)
    return df_C_T

def standarize_dataset(data):
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    data_new['X'] = X
    return data_new

def get_n_years(dataset, years):
    X, T, C, E, R = dataset['X'], dataset['T'], dataset['C'], dataset['E'], dataset['R']

    df = pd.DataFrame(X)
    df['T'] = T
    df['C'] = C
    df['R'] = R
    df['Y'] = 1

    df = df[~((df['T'] < 365 * years) & (df['C'] == 1))]
    df.loc[df['T'] <= 365 * years, 'Y'] = 0
    df['strat'] = df.apply(lambda row: str(row['Y']) + str(row['R']), axis=1)
    df = df.reset_index(drop=True)

    R = df['R'].values
    Y = df['Y'].values
    y_strat = df['strat'].values
    df = df.drop(columns=['T', 'C', 'R', 'Y', 'strat'])
    X = df.values
    y_sub = R # doese not matter

    return (X, Y.astype('int32'), R, y_sub, y_strat)

def run_cv(seed, fold, X, Y, R, y_strat, val_size=0, pretrain_set=None, batch_size=32, k=-1,
           learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, momentum=0.9,
           L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128,64]):

    X_w = pretrain_set.get_value(borrow=True) if k > 0 and pretrain_set else None

    m = X.shape[1] if k < 0 else k
    columns = list(range(m))
    columns.extend(['scr', 'pred','R', 'Y'])
    df = pd.DataFrame(columns=columns)
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X, y_strat):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        R_train, R_test = R[train_index], R[test_index]
        strat_train, strat_test = y_strat[train_index], y_strat[test_index]

        if k > 0:
            k_best = SelectKBest(f_classif, k=k)
            k_best.fit(X_train, Y_train)
            X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)

            if pretrain_set:
                X_base = k_best.transform(X_w)
                pretrain_set = theano.shared(X_base, name='pretrain_set', borrow=True)

        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                              test_size=val_size, random_state=0,
                                                              stratify=strat_train)
            valid_data = (X_val, Y_val)
        train_data = (X_train, Y_train)

        n_in = X_train.shape[1]
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers, momentum=momentum)
        if pretrain_set:
            pretrain_config = {'pt_batchsize': 32, 'pt_lr': 0.01, 'pt_epochs': 500, 'corruption_level': 0.3}
            classifier.pretrain(pretrain_set=pretrain_set, pretrain_config=pretrain_config)
            classifier.tune(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        else:
            classifier.train(train_data, valid_data=valid_data, batch_size=batch_size, n_epochs=n_epochs)
        X_scr = classifier.get_score(X_test)
        X_pred = classifier.get_pred(X_test)
        array1 = np.column_stack((X_test, X_scr[:,1],X_pred, R_test, Y_test))
        df_temp1 = pd.DataFrame(array1, index=list(test_index), columns=columns)
        df = df.append(df_temp1)

    return df

def run_mixture_cv(seed, dataset, fold=3, k=-1, val_size=0, batch_size=32, momentum=0.9,
                   learning_rate=0.01, lr_decay=0.0, dropout=0.5, n_epochs=100, save_to=None,
                   L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64], groups=("WHITE", "BLACK")):
    X, Y, R, y_sub, y_strat = dataset
    df = run_cv(seed, fold, X, Y, R, y_strat, val_size=val_size, batch_size=batch_size, k=k, momentum=momentum,
                learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout, n_epochs=n_epochs,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    if save_to:
        df.to_csv(save_to)
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
    y_test_w, y_scr_w = list(df.loc[df['R']==groups[0], 'Y'].values), \
                        list(df.loc[df['R']==groups[0], 'scr'].values)
    y_test_b, y_scr_b = list(df.loc[df['R']==groups[1], 'Y'].values), \
                        list(df.loc[df['R']==groups[1], 'scr'].values)

    A_CI, W_CI, B_CI = roc_auc_score(y_test, y_scr, average='weighted'), \
                       roc_auc_score(y_test_w, y_scr_w, average='weighted'), \
                       roc_auc_score(y_test_b, y_scr_b, average='weighted')

#    y_test, y_pred = list(df['Y'].values), list(df['pred'].values)
#    y_test_w, y_pred_w = list(df.loc[df['R']==groups[0], 'Y'].values), \
#                        list(df.loc[df['R']==groups[0], 'pred'].values)
#    y_test_b, y_pred_b = list(df.loc[df['R']==groups[1], 'Y'].values), \
#                        list(df.loc[df['R']==groups[1], 'pred'].values)
#    accuracy_A,precision_A,recall_A,f1_A = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), \
 #                                  recall_score(y_test, y_pred), f1_score(y_test, y_pred)
 #   accuracy_W,precision_W,recall_W,f1_W = accuracy_score(y_test_w, y_pred_w), precision_score(y_test_w, y_pred_w), \
 #                                  recall_score(y_test_w, y_pred_w), f1_score(y_test_w, y_pred_w)
#    accuracy_B,precision_B,recall_B,f1_B = accuracy_score(y_test_b, y_pred_b), precision_score(y_test_b, y_pred_b), \
#                                   recall_score(y_test_b, y_pred_b), f1_score(y_test_b, y_pred_b)
    res = {'folds': fold, 'A_Auc': A_CI,
           'W_Auc': W_CI, 'B_Auc': B_CI}
    df = pd.DataFrame(res, index=[seed])

 #   metrics = pd.DataFrame({
#    'folds':fold,'A_Acc':accuracy_A, 'A_Prec':precision_A, 'A_Recall':recall_A, 'A_F1':f1_A,
#    'W_Acc':accuracy_W, 'W_Prec':precision_W, 'W_Recall':recall_W, 'W_F1':f1_W,
#    'B_Acc':accuracy_B, 'B_Prec':precision_B, 'A_Recall':recall_B, 'A_F1':f1_B}, index=[seed])
 
    return df,y_scr,y_scr_w,y_scr_b

def run_one_race_cv(seed, dataset, fold=3,  k=-1, val_size=0, batch_size=32,
                    learning_rate=0.01, lr_decay=0.0, dropout=0.5, save_to=None,
                    L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64]):
    X, Y, R, y_sub, y_strat = dataset
    df = run_cv(seed, fold, X, Y, R, y_strat, val_size=val_size, batch_size=batch_size, k=k,
                learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hiddenLayers=hiddenLayers)
    if save_to:
        df.to_csv(save_to)
    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
#    y_test, y_pred = list(df['Y'].values), list(df['pred'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'folds': fold, 'Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
 #   accuracy,precision,recall,f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), \
 #                                  recall_score(y_test, y_pred), f1_score(y_test, y_pred)
 #   metrics = pd.DataFrame({
 #   'Ind_Acc':accuracy, 'Ind_Prec':precision, 'Ind_Recall':recall, 'Ind_F1':f1}, index=[seed])
    return df,y_scr

def run_supervised_transfer_cv(seed, dataset, fold=3, val_size=0, k=-1, batch_size=32, groups=('WHITE', 'BLACK'),
                    learning_rate=0.01, lr_decay=0.0, dropout=0.5, tune_epoch=200, tune_lr=0.002, train_epoch=1000,
                    L1_reg=0.001, L2_reg=0.001, hiddenLayers=[128, 64], tune_batch=10):
    X, Y, R, y_sub, y_strat = dataset
    idx = R == groups[1]
    X_b, y_b, R_b, y_strat_b = X[idx], Y[idx], R[idx], y_strat[idx]
    idx = R == groups[0]
    X_w, y_w, R_w, y_strat_w = X[idx], Y[idx], R[idx], y_strat[idx]
    pretrain_set = (X_w, y_w)

    df = pd.DataFrame(columns=['scr','pred', 'R', 'Y'])
    kf = StratifiedKFold(n_splits=fold, shuffle=True, random_state=seed)
    for train_index, test_index in kf.split(X_b, y_strat_b):
        X_train, X_test = X_b[train_index], X_b[test_index]
        Y_train, Y_test = y_b[train_index], y_b[test_index]
        R_train, R_test = R_b[train_index], R_b[test_index]
        strat_train, strat_test = y_strat_b[train_index], y_strat_b[test_index]

        if k > 0:
            k_best = SelectKBest(f_classif, k=k)
            k_best.fit(X_train, Y_train)
            X_train, X_test = k_best.transform(X_train), k_best.transform(X_test)
            X_base = k_best.transform(X_w)
            pretrain_set = (X_base, y_w)

        valid_data = None
        if val_size:
            X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train,
                                                              test_size=val_size, random_state=0,
                                                              stratify=strat_train)
            valid_data = (X_val, Y_val)
        train_data = (X_train, Y_train)

        n_in = X_train.shape[1]
        classifier = MLP(n_in=n_in, learning_rate=learning_rate, lr_decay=lr_decay, dropout=dropout,
                L1_reg=L1_reg, L2_reg=L2_reg, hidden_layers_sizes=hiddenLayers)
        classifier.train(pretrain_set, n_epochs=train_epoch, batch_size=batch_size)
        classifier.learning_rate = tune_lr
        classifier.tune(train_data, valid_data=valid_data, batch_size=tune_batch, n_epochs=tune_epoch)

        scr = classifier.get_score(X_test)
        X_pred = classifier.get_pred(X_test)
        array = np.column_stack((scr[:, 1], X_pred,R_test, Y_test))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'pred','R', 'Y'])
        df = df.append(df_temp)

    y_test, y_scr = list(df['Y'].values), list(df['scr'].values)
 #   y_test, y_pred = list(df['Y'].values), list(df['pred'].values)
    A_CI = roc_auc_score(y_test, y_scr)
    res = {'folds': fold, 'TL_Auc': A_CI}
    df = pd.DataFrame(res, index=[seed])
  #  accuracy,precision,recall,f1 = accuracy_score(y_test, y_pred), precision_score(y_test, y_pred), \
  #                                 recall_score(y_test, y_pred), f1_score(y_test, y_pred)
  #  metrics = pd.DataFrame({
  #  'XY_Acc':accuracy, 'XY_Prec':precision, 'XY_Recall':recall, 'XY_F1':f1}, index=[seed])
    return df,y_scr

def run_CCSA_transfer(seed, dataset, n_features, fold=3, alpha=0.25, learning_rate = 0.01,
                      hiddenLayers=[100, 50], dr=0.5, groups=("WHITE", "BLACK"),
                      momentum=0.0, decay=0, batch_size=32,
                      sample_per_class=2, repetition=1):
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(X)
    df['R'] = R
    df['Y'] = Y

    df_train = df[df['R'] == groups[0]]
    df_w_y = df_train['Y']
    df_train = df_train.drop(columns=['Y', 'R'])

    Y_train_source = df_w_y.values.ravel()
    X_train_source = df_train.values

    df_test = df[df['R'] == groups[1]]
    df_b_y = df_test['Y']
    df_test = df_test.drop(columns=['Y', 'R'])

    Y_test = df_b_y.values.ravel()
    X_test = df_test.values

    if n_features > 0 and n_features < X_test.shape[1]:
        X_train_source, X_test = get_k_best(X_train_source, Y_train_source, X_test, n_features)
    else:
        n_features = X_test.shape[1]

    df_score = pd.DataFrame(columns=['scr','Y','pred'])
    kf = StratifiedKFold(n_splits=fold,shuffle=True,random_state=seed)
    for train_index, test_index in kf.split(X_test, Y_test):
        X_train_target_full, X_test_target = X_test[train_index], X_test[test_index]
        Y_train_target_full, Y_test_target = Y_test[train_index], Y_test[test_index]

        index0 = np.where(Y_train_target_full == 0)
        index1 = np.where(Y_train_target_full == 1)

        target_samples = []
        target_samples.extend(index0[0][0:sample_per_class])
        target_samples.extend(index1[0][0:sample_per_class])

        X_train_target = X_train_target_full[target_samples]
        Y_train_target = Y_train_target_full[target_samples]

        X_val_target = [e for idx, e in enumerate(X_train_target_full) if idx not in target_samples]
        Y_val_target = [e for idx, e in enumerate(Y_train_target_full) if idx not in target_samples]
        X_val_target = np.array(X_val_target)
        Y_val_target = np.array(Y_val_target)
        best_score, best_Auc = train_and_predict(X_train_target, Y_train_target,
                                     X_train_source, Y_train_source,
                                     X_val_target, Y_val_target,
                                     X_test_target, Y_test_target,
                                     sample_per_class=sample_per_class,
                                     alpha=alpha, learning_rate=learning_rate,
                                     hiddenLayers=hiddenLayers, dr=dr,
                                     momentum=momentum, decay=decay,
                                     batch_size=batch_size,
                                     repetition=repetition,
                                     n_features=n_features)

        print (best_score.shape)
        print (Y_test_target.shape)
        pred = (best_score > 0.5).astype(int)
        array = np.column_stack((best_score, Y_test_target,pred))
        df_temp = pd.DataFrame(array, index=list(test_index), columns=['scr', 'Y','pred'])
        df_score = df_score.append(df_temp)

    auc = roc_auc_score(df_score['Y'].values, df_score['scr'].values)
    res = {'TL_Auc': auc}
    df = pd.DataFrame(res, index=[seed])
#    accuracy,precision,recall,f1 = accuracy_score(df_score['Y'].values, df_score['pred'].values), precision_score(df_score['Y'].values, df_score['pred'].values), \
#                                   recall_score(df_score['Y'].values, df_score['pred'].values), f1_score(df_score['Y'].values, df_score['pred'].values)
#    metrics = pd.DataFrame({
#    'CCSA_Acc':accuracy, 'CCSA_Prec':precision, 'CCSA_Recall':recall, 'CCSA_F1':f1}, index=[seed])
    return df,df_score['scr'].values

def train_and_predict(X_train_target, y_train_target,
                      X_train_source, y_train_source,
                      X_val_target, Y_val_target,
                      X_test, y_test,
                      repetition, sample_per_class,
                      alpha=0.25, learning_rate = 0.01,
                      hiddenLayers=[100, 50], dr=0.5,
                      momentum=0.0, decay=0, batch_size=32,
                      n_features = 400):
    # size of input variable for each patient
    domain_adaptation_task = 'WHITE_to_BLACK'
    input_shape = (n_features,)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # number of classes for digits classification
    nb_classes = 2
    # Loss = (1-alpha)Classification_Loss + (alpha)CSA
    alpha = alpha

    # Having two streams. One for source and one for target.
    model1 = Initialization.Create_Model(hiddenLayers=hiddenLayers, dr=dr)
    processed_a = model1(input_a)
    processed_b = model1(input_b)

    # Creating the prediction function. This corresponds to h in the paper.
    processed_a = Dropout(0.5)(processed_a)
    out1 = Dense(nb_classes)(processed_a)
    out1 = Activation('softmax', name='classification')(out1)

    distance = Lambda(Initialization.euclidean_distance, output_shape=Initialization.eucl_dist_output_shape,
                      name='CSA')([processed_a, processed_b])
    model = Model(inputs=[input_a, input_b], outputs=[out1, distance])
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate, momentum=momentum) # momentum=0., decay=0., decay=decay
    model.compile(loss={'classification': 'binary_crossentropy', 'CSA': Initialization.contrastive_loss},
                  optimizer=optimizer,
                  loss_weights={'classification': 1 - alpha, 'CSA': alpha})

    print('Domain Adaptation Task: ' + domain_adaptation_task)
    # for repetition in range(10):
    Initialization.Create_Pairs(domain_adaptation_task, repetition, sample_per_class,
                                X_train_target, y_train_target, X_train_source, y_train_source,
                                n_features=n_features)
    best_score, best_Auc = Initialization.training_the_model(model, domain_adaptation_task, repetition, sample_per_class,batch_size,
                                                             X_val_target, Y_val_target,
                                                             X_test, y_test)

    print('Best AUC for {} target sample per class and repetition {} is {}.'.format(sample_per_class,
                                                                                             repetition, best_Auc))
    return best_score, best_Auc

def normalize_dataset(data):
    X = data['X']
    data_new = {}
    for k in data:
        data_new[k] = data[k]
    X = preprocessing.normalize(X)
    data_new['X'] = X
    return data_new

def run_PRAD_PFI_cv(data,years):
    dataset = data
    if dataset['X'].shape[0] < 10: return None
    dataset = standarize_dataset(dataset)
    dataset_w = get_one_race(dataset, 'WHITE')

    dataset_w = get_n_years(dataset_w, years)
    dataset_b = get_one_race(dataset, 'BLACK')

    dataset_b = get_n_years(dataset_b, years)

    dataset_tl = normalize_dataset(dataset)
    dataset_tl = get_n_years(dataset_tl, years)

    dataset = get_n_years(dataset, years)
    
    k = 200
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(y_strat, columns=['RY'])
    df['R'] = R
    df['Y'] = Y
    print(X.shape)
    Dict = df['RY'].value_counts()

    Dict = dict(Dict)
    print (Dict)
    for key in Dict:
        print (key, Dict[key])
     
    parametrs_mix = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,'momentum':0.9,
                     'learning_rate':0.01, 'lr_decay':0.03, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_w = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_b = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':4,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}

    parametrs_tl = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20, 'tune_epoch':500,
                    'learning_rate':0.01, 'lr_decay':0.03, 'dropout':0.5, 'tune_lr':0.002,
                    'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64], 'tune_batch':10}

    parametrs_tl_unsupervised = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
                     'learning_rate':0.001, 'lr_decay':0.03, 'dropout':0.0, 'n_epochs':100,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}
    parameters_CCSA = {'fold': 3, 'n_features': k, 'alpha':0.3, 'batch_size':20, 'learning_rate':0.01,
                       'hiddenLayers': [100], 'dr':0.0, 'momentum':0.9,
                       'decay':0.0, 'sample_per_class':2}
    
    res = pd.DataFrame()
    score_dict = {}
    for i in range(20):
        seed = i
        df_m,s_a,s_w,s_b = run_mixture_cv(seed, dataset, **parametrs_mix)
        df_w,s_w1 = run_one_race_cv(seed, dataset_w, **parametrs_w)
        df_w = df_w.rename(columns={"Auc": "W_ind"})
        df_b,s_b1 = run_one_race_cv(seed, dataset_b, **parametrs_b)
        df_b = df_b.rename(columns={"Auc": "B_ind"})
        df_tl_supervised,s_xy = run_supervised_transfer_cv(seed, dataset, **parametrs_tl)
        df_tl_supervised = df_tl_supervised.rename(columns={"TL_Auc": "XY_TL"})

        #   df_tl_unsupervised = run_unsupervised_transfer_cv(seed, dataset, **parametrs_tl_unsupervised)
        #   df_tl_unsupervised = df_tl_unsupervised.rename(columns={"TL_Auc": "X_TL"})

        df_tl,s_ccsa = run_CCSA_transfer(seed, dataset_tl, **parameters_CCSA)
        df_tl = df_tl.rename(columns={"TL_Auc": "CCSA_TL"})
    #df_tl_unsupervised['X_TL'],
        df1 = pd.concat([df_m, df_w['W_ind'], df_b['B_ind'], df_tl['CCSA_TL'],
                            df_tl_supervised['XY_TL']],
                            sort=False, axis=1)
        s_a_df = pd.DataFrame(s_a)
        s_w_df = pd.DataFrame(s_w)
        s_b_df = pd.DataFrame(s_b)
        s_w1_df = pd.DataFrame(s_w1)
        s_b1_df = pd.DataFrame(s_b1)
        s_xy_df = pd.DataFrame(s_xy)
        s_ccsa_df = pd.DataFrame(s_ccsa)
        all_s = pd.concat([s_a_df,s_w_df,s_b_df,s_w1_df,s_b1_df,s_xy_df,s_ccsa_df],sort=False, axis=1)
        res = res.append(df1)
        score_dict[i] = all_s
        
    f_name = 'PRAD-AA-EA-miRNA-PFI-' + str(years) + 'YR.xlsx'
    res.to_excel(f_name)
    summary_df = pd.DataFrame({'Column': res.columns,
                           'Mean': res.mean(),
                           'Standard Deviation': res.std()})
    summary_df.to_excel('summary-PRAD-miRNA-PFI-'+ str(years) + 'YR.xlsx')
    with open('score_dict_exp.pkl', 'wb') as file:
        pickle.dump(score_dict, file)




def run_PRAD_PFI_inter_cv(data,years):
    dataset = data
    if dataset['X'].shape[0] < 10: return None
    dataset = standarize_dataset(dataset)
    dataset_w = get_one_race(dataset, 'WHITE')

    dataset_w = get_n_years(dataset_w, years)
    dataset_b = get_one_race(dataset, 'BLACK')

    dataset_b = get_n_years(dataset_b, years)

    dataset_tl = normalize_dataset(dataset)
    dataset_tl = get_n_years(dataset_tl, years)

    dataset = get_n_years(dataset, years)
    
    k = 400
    X, Y, R, y_sub, y_strat = dataset
    df = pd.DataFrame(y_strat, columns=['RY'])
    df['R'] = R
    df['Y'] = Y
    print(X.shape)
    Dict = df['RY'].value_counts()

    Dict = dict(Dict)
    print (Dict)
    for key in Dict:
        print (key, Dict[key])
     
    parametrs_mix = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,'momentum':0.9,
                     'learning_rate':0.01, 'lr_decay':0.03, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_w = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}
    parametrs_b = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':4,
                     'learning_rate':0.01, 'lr_decay':0.0, 'dropout':0.5,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64]}

    parametrs_tl = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20, 'tune_epoch':500,
                     'learning_rate':0.01, 'lr_decay':0.03, 'dropout':0.5, 'tune_lr':0.002,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [128, 64], 'tune_batch':10}

    parametrs_tl_unsupervised = {'fold': 3, 'k': k, 'val_size':0.0, 'batch_size':20,
                     'learning_rate':0.001, 'lr_decay':0.03, 'dropout':0.0, 'n_epochs':100,
                     'L1_reg': 0.001, 'L2_reg': 0.001, 'hiddenLayers': [100]}
    parameters_CCSA = {'fold': 3, 'n_features': k, 'alpha':0.3, 'batch_size':20, 'learning_rate':0.01,
                       'hiddenLayers': [100], 'dr':0.0, 'momentum':0.9,
                       'decay':0.0, 'sample_per_class':2}
    
    res = pd.DataFrame()
    score_dict = {}
    for i in range(20):
        seed = i
        df_m,s_a,s_w,s_b = run_mixture_cv(seed, dataset, **parametrs_mix)
        df_w,s_w1 = run_one_race_cv(seed, dataset_w, **parametrs_w)
        df_w = df_w.rename(columns={"Auc": "W_ind"})
        df_b,s_b1 = run_one_race_cv(seed, dataset_b, **parametrs_b)
        df_b = df_b.rename(columns={"Auc": "B_ind"})
        df_tl_supervised,s_xy = run_supervised_transfer_cv(seed, dataset, **parametrs_tl)
        df_tl_supervised = df_tl_supervised.rename(columns={"TL_Auc": "XY_TL"})
    
         #   df_tl_unsupervised = run_unsupervised_transfer_cv(seed, dataset, **parametrs_tl_unsupervised)
         #   df_tl_unsupervised = df_tl_unsupervised.rename(columns={"TL_Auc": "X_TL"})
    
        df_tl,s_ccsa = run_CCSA_transfer(seed, dataset_tl, **parameters_CCSA)
        df_tl = df_tl.rename(columns={"TL_Auc": "CCSA_TL"})
     #df_tl_unsupervised['X_TL'],
        df1 = pd.concat([df_m, df_w['W_ind'], df_b['B_ind'], df_tl['CCSA_TL'],
                             df_tl_supervised['XY_TL']],
                            sort=False, axis=1)
        s_a_df = pd.DataFrame(s_a)
        s_w_df = pd.DataFrame(s_w)
        s_b_df = pd.DataFrame(s_b)
        s_w1_df = pd.DataFrame(s_w1)
        s_b1_df = pd.DataFrame(s_b1)
        s_xy_df = pd.DataFrame(s_xy)
        s_ccsa_df = pd.DataFrame(s_ccsa)
        all_s = pd.concat([s_a_df,s_w_df,s_b_df,s_w1_df,s_b1_df,s_xy_df,s_ccsa_df],sort=False, axis=1)
        res = res.append(df1)
        score_dict[i] = all_s
        #m_res = m_res.append(all_m)

    f_name = 'PRAD-AA-EA-integration_mRNA_Methy_VAE-PFI-' + str(years) + 'YR.xlsx'
    res.to_excel(f_name)
    summary_df = pd.DataFrame({'Column': res.columns,
                           'Mean': res.mean(),
                           'Standard Deviation': res.std()})
    summary_df.to_excel('summary-PRAD-integration_mRNA_Methy_VAE-PFI-'+ str(years) + 'YR.xlsx')
    name = 'PRAD-AA-EA-integration_mRNA_Methy_VAE-PFI-' + str(years) + 'YR.pkl'
    with open(name, 'wb') as file:
        pickle.dump(score_dict, file)

inter_m_me_VAE = pd.read_csv('inte_mRNA_Methy_PRAD_VAE.csv')
inter_m_me_VAE = inter_m_me_VAE.set_index('index')
inter_m_me_PRAD_VAE = add_race_CT(cancer_type = 'PRAD', df = inter_m_me_VAE, target = 'PFI', groups = ("WHITE", "BLACK"))

run_PRAD_PFI_inter_cv(data = inter_m_me_PRAD_VAE,years = 3)
run_PRAD_PFI_inter_cv(data = inter_m_me_PRAD_VAE,years = 2)
run_PRAD_PFI_inter_cv(data = inter_m_me_PRAD_VAE,years = 4)
run_PRAD_PFI_inter_cv(data = inter_m_me_PRAD_VAE,years = 5)