import tensorflow as tf
import os
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

from tensorflow.keras import regularizers
import numpy as np
from time import time
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *

from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense

np.set_printoptions(threshold=np.inf)

# cifar10 = tf.keras.datasets.cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
X_train = pd.read_csv('./data/dataset03.csv',usecols=['F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3','L_T1'])
X_test = pd.read_csv('./data/test_dataset.csv',usecols=['F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3','L_T1'])
act_func = 'relu'
# X_train = [X_train.L_T1,[X_train.F_PU1,X_train.F_PU2,X_train.F_PU3,[X_train.S_PU1,X_train.S_PU2,X_train.S_PU3]]]
# X_test = [X_test.L_T1,[X_test.F_PU1,X_test.F_PU2,X_test.F_PU3,[X_test.S_PU1,X_test.S_PU2,X_test.S_PU3]]]
model = Sequential()

TIME_PERIODS = 7
input_shape=(TIME_PERIODS,)
model.add(Reshape((TIME_PERIODS,1), input_shape=input_shape))
model.add(Conv1D(8, 4, strides=2, activation='relu', input_shape=(TIME_PERIODS, 1)))

model.add(Conv1D(8, 4, strides=2, activation='relu', padding="same"))
model.add(MaxPooling1D(1))

model.add(Conv1D(2, 4, strides=2, activation='relu', padding="same"))
model.add(Conv1D(10, 4, strides=2, activation='relu', padding="same"))
model.add(MaxPooling1D(1))
model.add(Conv1D(2, 4, strides=2, activation='relu', padding="same"))
model.add(Conv1D(10, 4, strides=2, activation='relu', padding="same"))
model.add(MaxPooling1D(1))
model.add(Conv1D(2, 2, strides=1, activation='relu', padding="same"))
model.add(Conv1D(10, 2, strides=1, activation='relu', padding="same"))
model.add(MaxPooling1D(1))
"""model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))"""
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.3))

model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(X_train.shape[1],)
               )
         )

model.add(Dense(2,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(X_train.shape[1],
                kernel_initializer='glorot_uniform'))

# model.add(Dense(7, activation='softmax'))

model.compile(loss='mse',optimizer='adam')

history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=10,
                  epochs=50,
                  validation_split=0.05,
                  verbose = 1)
# history = model.fit(np.array(X_train), np.array(X_train), batch_size=32, epochs=300, validation_data=(np.array(X_test), np.array(X_test)), validation_freq=1
#                     )
print(model.summary())

plt.plot(history.history['loss'],
         'b',
         label='Training loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.show()
X_pred = model.predict(np.array(X_train))
X_pred = pd.DataFrame(X_pred,
                      columns=X_train.columns)
print(X_pred)
X_pred.index = X_train.index
scored = pd.DataFrame(index=X_train.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_train), axis = 1)
plt.figure()
sns.distplot(scored['Loss_mae'],
             bins = 10,
             kde= True,
            color = 'blue')
plt.xlim([0.0,.5])
plt.show()
X_pred = model.predict(np.array(X_test))
X_pred = pd.DataFrame(X_pred,
                      columns=X_test.columns)
X_pred.index = X_test.index
threshod = 0.4
scored = pd.DataFrame(index=X_test.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
scored['Threshold'] = threshod
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()
scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','green'])
plt.show()

