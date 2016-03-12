from __future__ import print_function

import numpy as np
# from keras.models import Sequential
from keras.models import Graph
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K
from keras.layers.normalization import BatchNormalization
# from keras import activations, initializations, regularizers, constraints
# from keras.regularizers import ActivityRegularizer

def center_normalize(x):
    """
    Custom activation for online sample-wise center and std. normalization
    """
    return (x - K.mean(x)) / K.std(x)

def get_model():
    graph = Graph()
    graph.add_input(name='input1', input_shape=(30,64,64)) #_,128,128
    graph.add_input(name='input2', input_shape=(2,)) # 2,498
    graph.add_node(Activation(activation=center_normalize), name='center_normalize', input='input1')
    
    graph.add_node(Convolution2D(64, 3, 3, border_mode='same'), name='Convolution2D1_1', input='center_normalize')
    graph.add_node(Activation('relu'), name='relu1_1', input='Convolution2D1_1')
    graph.add_node(Convolution2D(64, 3, 3, border_mode='valid'), name='Convolution2D1_2', input='relu1_1')
    graph.add_node(Activation('relu'), name='relu1_2', input='Convolution2D1_2')
    graph.add_node(ZeroPadding2D(padding=(1, 1)), name='zeropad1', input='relu1_2')
    graph.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), name='maxpool1', input='zeropad1')
    graph.add_node(Dropout(0.25), name='dropout1', input='maxpool1')

    graph.add_node(Convolution2D(64, 3, 3, border_mode='same'), name='Convolution2D2_1', input='dropout1')
    graph.add_node(Activation('relu'), name='relu2_1', input='Convolution2D2_1')
    graph.add_node(Convolution2D(64, 3, 3, border_mode='valid'), name='Convolution2D2_2', input='relu2_1')
    graph.add_node(Activation('relu'), name='relu2_2', input='Convolution2D2_2')
    graph.add_node(ZeroPadding2D(padding=(1, 1)), name='zeropad2', input='relu2_2')
    graph.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), name='maxpool2', input='zeropad2')
    graph.add_node(Dropout(0.25), name='dropout2', input='maxpool2')

    graph.add_node(Convolution2D(64, 3, 3, border_mode='same'), name='Convolution2D3_1', input='dropout2')
    graph.add_node(Activation('relu'), name='relu3_1', input='Convolution2D3_1')
    graph.add_node(Convolution2D(64, 3, 3, border_mode='valid'), name='Convolution2D3_2', input='relu3_1')
    graph.add_node(Activation('relu'), name='relu3_2', input='Convolution2D3_2')
    graph.add_node(ZeroPadding2D(padding=(1, 1)), name='zeropad3', input='relu3_2')
    graph.add_node(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)), name='maxpool3', input='zeropad3')
    graph.add_node(Dropout(0.25), name='dropout3', input='maxpool3')

    graph.add_node(Flatten(), name='flatten', input='dropout3')
    graph.add_node(Dense(1024, W_regularizer=l2(1e-3)), name='dense1', inputs=['flatten', 'input2'], merge_mode='concat', concat_axis=1)
    graph.add_node(Activation('relu'), name='relu4', input='dense1')
    graph.add_node(Dense(1), name='output', input='relu4', create_output=True)

    adam = Adam(lr=0.0001)
    graph.compile(optimizer=adam, loss={'output':'mse'})
    return graph
