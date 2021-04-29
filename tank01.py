import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
import numpy as np
from time import time

import seaborn as sns
sns.set(color_codes=True)
startTime1 = time()
X_train = pd.read_csv('./data/dataset03.csv',usecols=['L_T1','F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3','P_J280','P_J269','P_J300','P_J256','P_J289','P_J415','P_J302','P_J306','P_J307','P_J317','P_J14','P_J422'])
X_test = pd.read_csv('./data/test_dataset.csv',usecols=['L_T1','F_PU1','S_PU1','S_PU2','F_PU2','S_PU3','F_PU3','P_J280','P_J269','P_J300','P_J256','P_J289','P_J415','P_J302','P_J306','P_J307','P_J317','P_J14','P_J422'])

act_func = 'relu'
# Input layer:
model=Sequential()
# First hidden layer, connected to input vector X.
model.add(Dense(21,activation=act_func,
                kernel_initializer='glorot_uniform',
                kernel_regularizer=regularizers.l2(0.0),
                input_shape=(X_train.shape[1],)
               )
         )

model.add(Dense(8,activation=act_func,
                kernel_initializer='glorot_uniform'))

model.add(Dense(21,activation=act_func,
                kernel_initializer='glorot_uniform'))
model.add(Dense(8,activation=act_func,
                kernel_initializer='glorot_uniform'))
model.add(Dense(X_train.shape[1],
                kernel_initializer='glorot_uniform'))
model.compile(loss='mse',optimizer='adam')

print(model.summary())

# Train model for 100 epochs, batch size of 10:
NUM_EPOCHS=10
BATCH_SIZE=10
history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=BATCH_SIZE,
                  epochs=NUM_EPOCHS,
                  validation_split=0.05,
                  verbose = 1)

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
threshod = 0.5
scored = pd.DataFrame(index=X_test.index)
scored['Loss_mae'] = np.mean(np.abs(X_pred-X_test), axis = 1)
scored['Threshold'] = threshod
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()
scored.plot(logy=True,  figsize = (10,6), ylim = [1e-2,1e2], color = ['blue','green'])
plt.show()
t1 = time() - startTime1

print(t1)