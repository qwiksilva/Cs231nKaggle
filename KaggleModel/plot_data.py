from __future__ import print_function

import sys
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os.path

from model import get_model
from utils import crps, real_to_cdf, preprocess, rotation_augmentation, shift_augmentation, root_mean_squared_error
import matplotlib.pyplot as plt
import csv

def load_train_data():
    """
    Load training data from .npy files.
    """
    X = np.load('/data/preprocessed/X_train.npy')
    y = np.load('/data/preprocessed/y_train.npy')
    metadata = np.load('/data/preprocessed/metadata_train.npy')

    X = X[:, :30*15, :, :]
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
# txt_file = r'/data/run2/loss.txt'
# csv_file = r"mycsv.csv"

# # use 'with' if the program isn't going to immediately terminate
# # so you don't leave files open
# # the 'b' is necessary on Windows
# # it prevents \x1a, Ctrl-z, from ending the stream prematurely
# # and also stops Python converting to / from different line terminators
# # On other platforms, it has no effect
# in_txt = csv.reader(open(txt_file, "rb"), delimiter = '\t')
# out_csv = csv.writer(open(csv_file, 'wb'))

# out_csv.writerows(in_txt)
# spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
# 	for row in spamreader:
# 		print ', '.join(row)

# # Loss
# plt.plot(x,y)
# plt.xlabel('Iteration')
# plt.ylabel('RMSE')
# plt.title('RMSE Loss Function')
# plt.legend('')
# plt.show()

# #crps
# plt.plot(x,y)
# plt.xlabel('Iteration')
# plt.ylabel('CRPS')
# plt.title('')
# plt.legend('')
# plt.show()


print('Loading and compiling models...')
model_systole = get_model()
model_diastole = get_model()

#import best model if it exists
if os.path.isfile('/data/run2/weights_systole_best.hdf5'):
    print('loading weights')
    model_systole.load_weights('/data/run2/weights_systole_best.hdf5')

if os.path.isfile('/data/run2/weights_diastole_best.hdf5'):
    model_diastole.load_weights('/data/run2/weights_diastole_best.hdf5')

print('Loading training data...')
X, y, metadata = load_train_data()

pred_systole = model_systole.predict({'input1':X[0], 'input2':metadata[0], 'output':y[0, 0]})['output']
pred_diastole = model_diastole.predict({'input1':X[0], 'input2':metadata[0], 'output':y[0, 1]})['output']

# CDF for train and test data (actually a step function)
cdf_train = real_to_cdf(np.concatenate((y[0, 0], y[0, 1])))

# CDF for predicted data
cdf_pred_systole = real_to_cdf(pred_systole, loss_systole)
cdf_pred_diastole = real_to_cdf(pred_diastole, loss_diastole)

# evaluate CRPS on training data
crps_train = crps(cdf_train, np.concatenate((cdf_pred_systole, cdf_pred_diastole)))
print('CRPS(train) = {0}'.format(crps_train))

# CDF
#plt.figure()
plt.plot(cdf_train[0], 'g-')
plt.plot(cdf_pred_systole, 'b-')
plt.xlabel('Volume')
plt.ylabel('Probability')
plt.title('Systole CDF vs Ground Truth (CRPS: ' + str(crps_train) + ")")
plt.legend(['Ground Truth CDF', 'Predicted Systole CDF'])
plt.savefig('systole.png', format='png')

#plt.figure()
plt.plot(cdf_train[1], 'g-')
plt.plot(cdf_pred_diastole, 'b-')
plt.xlabel('Volume')
plt.ylabel('Probability')
plt.title('Diastole CDF vs Ground Truth (CRPS: ' + str(crps_train) + ")")
plt.legend(['Ground Truth CDF', 'Predicted Diastole CDF'])
plt.show()
plt.savefig('diastole.png', format='png')