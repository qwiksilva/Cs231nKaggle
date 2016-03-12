from __future__ import print_function

import numpy as np
import keras.layers
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras import activations, initializations, regularizers, constraints
from keras.regularizers import ActivityRegularizer

def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)

class Dense2(keras.layers.Layer):
    input_ndim = 2

    def __init__(self, extra_inputs, batch_size, size, output_dim, init='glorot_uniform', activation='linear', weights=None,
                 W_extra_regularizer=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.counter = 0

        self.nb_batch = int(np.ceil(size / float(batch_size)))
        self.batches = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, self.nb_batch)]

        self.extra_inputs = extra_inputs
        self.extra_dim = self.extra_inputs.shape[1]

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.output_dim = output_dim

        self.W_extra_regularizer = regularizers.get(W_extra_regularizer)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.constraints = [self.W_constraint, self.b_constraint]

        self.initial_weights = weights

        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Dense2, self).__init__(**kwargs)

    def build(self):
        input_dim = self.input_shape[1]
        input_dim += self.extra_dim

        self.W_extra = self.init((input_dim, self.extra_dim),
                           name='{}_W_extra'.format(self.name))
        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        self.b = K.zeros((self.output_dim,),
                         name='{}_b'.format(self.name))

        self.trainable_weights = [self.W, self.W_extra, self.b]

        self.regularizers = []
        if self.W_extra_regularizer:
            self.W_extra_regularizer.set_param(self.W_extra)
            self.regularizers.append(self.W_extra_regularizer)

        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    @property
    def output_shape(self):
        return (self.input_shape[0], self.output_dim)

    def get_output(self, train=False):
        (batch_start, batch_end) = self.batches[self.counter]

        X = self.get_input(train)
	if X.shape[0] != self.extra_inputs.shape[0]:
	    output = self.activation(K.dot(X, self.W) + self.b)
	else:
            X_concat = np.concatenate((X, self.extra_inputs[batch_start:batch_end, :]), axis=1)
            W_concat = np.concatenate((self.W, self.W_extra), axis=0)
	    output = self.activation(K.dot(X_concat, W_concat) + self.b)

        if self.counter == self.nb_batch:
            self.counter = 0
        else:
            self.counter += 1

        return output

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def get_model():
    model = Sequential()
    model.add(Activation(activation=center_normalize, input_shape=(30, 64, 64)))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))


    model.add(Convolution2D(96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))

    model.add(Convolution2D(96, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))


    model.add(Convolution2D(64, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))

    model.add(Convolution2D(64, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    #model.add(BatchNormalization(axis=1))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense2(np.random.rand(4265, 2), 32, 500, 1024, W_extra_regularizer=l2(1e-3), W_regularizer=l2(1e-3)))
    #model.add(Dense(1024, W_regularizer=l2(1e-3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss='mse')
    return model
