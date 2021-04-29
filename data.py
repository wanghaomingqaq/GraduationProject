import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('./data/dataset03.csv',nrows=50)
data2 = pd.read_csv('./data/dataset04.csv',index_col='DATETIME')
data2_X= data2['10/09/16 02':'19/09/16 10'] # 13/09/16 23   16/09/16 00

print(data)
# plt.plot(data.S_PU8, color = 'black')
# plt.plot(data.S_PU11, color = 'blue')
plt.plot(data.S_PU10, color = 'green')
# plt.plot(data.S_PU11,color='yellow')
plt.plot(data.L_T7, color = 'red')
# plt.plot(data2.S_PU2, color = 'green')
# plt.plot(data2.L_T7,color = 'red')
# plt.plot(data2_X.F_PU9)
# plt.axvline('13/09/16 23',color='red')
# plt.axvline('16/09/16 00',color='red')
plt.show()
