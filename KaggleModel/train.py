from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os.path

from model import get_model
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation, root_mean_squared_error
from theano.tensor import as_tensor_variable

def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('/data/preprocessed/X_train.npy')
    y = np.load('/data/preprocessed/y_train.npy')
    metadata = np.load('/data/preprocessed/metadata_train.npy')

    X = X.astype(np.float32)
    X /= 255

    seed = 12345
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    np.random.seed(seed)
    np.random.shuffle(metadata)

    return X, y, metadata


def split_data(X, y, metadata, split_ratio=0.2):
    """
    Split data into training and testing.

    :param X: X
    :param y: y
    :param split_ratio: split ratio for train and test data
    """
    split = X.shape[0] * split_ratio
    X_test = X[:split, :, :, :]
    y_test = y[:split, :]
    metadata_test = metadata[:split, :]

    X_train = X[split:, :, :, :]
    y_train = y[split:, :]
    metadata_train = metadata[split:, :]
    
    return X_train, y_train, X_test, y_test, metadata_train, metadata_test


def train():
    """
    Training systole and diastole models.
    """
    print('Loading and compiling models...')
    model_systole = get_model()
    model_diastole = get_model()

    #import best model if it exists
    if os.path.isfile('/data/run/weights_systole_best.hdf5'):
        print('loading weights')
        model_systole.load_weights('/data/run/weights_systole_best.hdf5')

    if os.path.isfile('/data/run/weights_diastole_best.hdf5'):
        model_diastole.load_weights('/data/run/weights_diastole_best.hdf5')

    print('Loading training data...')
    X, y, metadata = load_train_data()

    #print('Pre-processing images...')
    #X = preprocess(X)
    #np.save('/data/pre/pre/X_train.npy', X)


    # split to training and test
    X_train, y_train, X_test, y_test, metadata_train, metadata_test = split_data(X, y, metadata, split_ratio=0.2)

    nb_iter = 200
    epochs_per_iter = 1
    batch_size = 8
    calc_crps = 5  # calculate CRPS every n-th iteration (set to 0 if CRPS estimation is not needed)

    # remember min val. losses (best iterations), used as sigmas for submission
    min_val_loss_systole = sys.float_info.max
    min_val_loss_diastole = sys.float_info.max

    print('-'*50)
    print('Training...')
    print('-'*50)

    for i in range(0,nb_iter):
        print('-'*50)
        print('Iteration {0}/{1}'.format(i + 1, nb_iter))
        print('-'*50)

        # print('Augmenting images - rotations')
        # X_train_aug = rotation_augmentation(X_train, 15)
        # print('Augmenting images - shifts')
        # X_train_aug = shift_augmentation(X_train_aug, 0.1, 0.1)
        X_train_aug = X_train

        print('Fitting systole model...')
        hist_systole = model_systole.fit({'input1':X_train_aug, 'input2':metadata_train, 'output':y_train[:, 0]}, shuffle=True, nb_epoch=epochs_per_iter,
                                         batch_size=batch_size, validation_data={'input1':X_test,'input2':metadata_test, 'output':y_test[:, 0]})

        print('Fitting diastole model...')
        hist_diastole = model_diastole.fit({'input1':X_train_aug, 'input2':metadata_train, 'output':y_train[:, 1]}, shuffle=True, nb_epoch=epochs_per_iter,
                                           batch_size=batch_size, validation_data={'input1':X_test, 'input2':metadata_test, 'output':y_test[:, 1]})

        # sigmas for predicted data, actually loss function values (RMSE)
        loss_systole = hist_systole.history['loss'][-1]
        loss_diastole = hist_diastole.history['loss'][-1]
        val_loss_systole = hist_systole.history['val_loss'][-1]
        val_loss_diastole = hist_diastole.history['val_loss'][-1]

        if calc_crps > 0 and i % calc_crps == 0:
            print('Evaluating CRPS...')
            pred_systole = model_systole.predict({'input1':X_train, 'input2':metadata_train, 'output':y_train[:, 0]}, batch_size=batch_size, verbose=1)['output']
            pred_diastole = model_diastole.predict({'input1':X_train, 'input2':metadata_train, 'output':y_train[:, 1]}, batch_size=batch_size, verbose=1)['output']
            val_pred_systole = model_systole.predict({'input1':X_test, 'input2':metadata_test, 'output':y_test[:, 0]}, batch_size=batch_size, verbose=1)['output']
            val_pred_diastole = model_diastole.predict({'input1':X_test, 'input2':metadata_test, 'output':y_test[:, 1]}, batch_size=batch_size, verbose=1)['output']

            # Get sigmas
            # sigma_systole = as_tensor_variable(root_mean_squared_error(y_train[:, 0], pred_systole))
            # sigma_diastole = as_tensor_variable(root_mean_squared_error(y_train[:, 1], pred_systole))
            # val_sigma_systole = as_tensor_variable(root_mean_squared_error(y_test[:, 0], val_pred_systole))
            # val_sigma_diastole = as_tensor_variable(root_mean_squared_error(y_test[:, 1], val_pred_diastole))

            # CDF for train and test data (actually a step function)
            cdf_train = real_to_cdf(np.concatenate((y_train[:, 0], y_train[:, 1])))
            cdf_test = real_to_cdf(np.concatenate((y_test[:, 0], y_test[:, 1])))

            # CDF for predicted data
            cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
            cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)
            cdf_val_pred_systole = real_to_cdf(val_pred_systole, val_loss_systole)
            cdf_val_pred_diastole = real_to_cdf(val_pred_diastole, val_loss_diastole)

            # evaluate CRPS on training data
            crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
            print('CRPS(train) = {0}'.format(crps_train))

            # evaluate CRPS on test data
            crps_test = crps(cdf_test, np.concatenate((cdf_val_pred_systole, cdf_val_pred_diastole)))
            print('CRPS(test) = {0}'.format(crps_test))

        # for best (lowest) val losses, save weights
        if val_loss_systole < min_val_loss_systole:
            min_val_loss_systole = val_loss_systole
            model_systole.save_weights('/data/run/weights_systole_best.hdf5', overwrite=True)

        if val_loss_diastole < min_val_loss_diastole:
            min_val_loss_diastole = val_loss_diastole
            model_diastole.save_weights('/data/run/weights_diastole_best.hdf5', overwrite=True)

        # save best (lowest) val losses in file (to be later used for generating submission)
        with open('/data/run/val_loss.txt', mode='w+') as f:
            f.write(str(min_val_loss_systole))
            f.write('\n')
            f.write(str(min_val_loss_diastole))

        with open("/data/run/loss.txt", "a+") as myfile:
            myfile.write('\t'.join((str(i+1), str(loss_systole),str(loss_diastole),str(val_loss_systole),str(val_loss_diastole), str(crps_train), str(crps_test))))
            myfile.write('\n')


train()
