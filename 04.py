import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import numpy as np
from time import time
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras import Model

np.set_printoptions(threshold=np.inf)

# cifar10 = tf.keras.datasets.cifar10
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
X_train = pd.read_csv('./data/dataset03.csv',usecols=['F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3','L_T1'])
X_test = pd.read_csv('./data/test_dataset.csv',usecols=['F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3','L_T1'])
act_func = 'relu'
# X_train = [X_train.L_T1,[X_train.F_PU1,X_train.F_PU2,X_train.F_PU3,[X_train.S_PU1,X_train.S_PU2,X_train.S_PU3]]]
# X_test = [X_test.L_T1,[X_test.F_PU1,X_test.F_PU2,X_test.F_PU3,[X_test.S_PU1,X_test.S_PU2,X_test.S_PU3]]]

print(X_train)
model=Sequential()

model.add(Conv1D(filters=6,
                 kernel_size=3,
                 padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Dropout(0.2))


model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(7,)
               )
         )

model.add(Dense(2,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(10,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(7,
                kernel_initializer='glorot_uniform'))

model.compile(loss='mse',optimizer='adam')
history = model.fit(np.array(X_train), np.array(X_train), batch_size=32, epochs=5, validation_data=(np.array(X_test), np.array(X_test)), validation_freq=1
                    )
print(model.summary())