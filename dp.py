# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 23:28:42 2020

@author: Administrator
"""

import pandas as pd
import numpy as np
from sklearn import metrics
from keras.models import Sequential
from keras.layers import LSTM,GRU
from keras.layers import Dense,Activation
from keras.layers import Dropout
import matplotlib.pyplot as plt

data = pd.read_excel('/data1.xlsx',encoding='gb18030')
# 将5个城市天气指标数据类型转换成category
samples=data.iloc[:,3:8].values
token_index={}
for sample in samples:
    for word in sample:
        if word not in token_index and word is not np.nan :
            token_index[word] = len(token_index)+1

for i in range(data.shape[0]):
    for j in range(3,8):
        c=data.iloc[i,j]
        data.iloc[i,j]=token_index.get(c)
        
        
X_data = data[['日期', '天平均运输价格（分/吨公里）', 'changjiang', 'loudi', 'xiangtan', 'yueyang',
       'yangchun']]# 提取特征数据
Y_data = data[['货运量']] #提取货运量数据

def get_data(X_data,Y_data):
    X = []
    Y = []
    X_data=X_data.values
    Y_data=Y_data.values
    row_length = len(X_data)
    
    for i in range(row_length):
        
        X.append(X_data[i].tolist())
        Y.append(Y_data[i].tolist())
    
    input_X = np.array(X).reshape(-1,7)
    input_Y = np.array(Y)

    return input_X,input_Y

a,b=get_data(X_data,Y_data)
ori_data=np.concatenate((a,b),axis=1)
row_length=ori_data.shape[0]

#构建输入 cat_X是未归一化的输入，代表包括i时刻+向前推step-1天的外部影响因素指标+对应的货运量
#构建目标值cat_Y是模型输出，代表i+1时刻的货运量
step=5
cat_X=[]
cat_Y=[]
for i in range(row_length-step):
    cat_X.append(ori_data[i:i+step,:].flatten().tolist())
    cat_Y.append(ori_data[i+step,-1].tolist())


X=np.array(cat_X)
Y=np.array(cat_Y).reshape((-1,1))

Y_val = Y[int(X.shape[0] * 0.8):]


from sklearn.preprocessing import MinMaxScaler

scaler_for_x=MinMaxScaler(feature_range=(0,1))  #按列做minmax缩放
scaler_for_y=MinMaxScaler(feature_range=(0,1))
scaled_x_data=scaler_for_x.fit_transform(X)
scaled_y_data=scaler_for_y.fit_transform(Y)

X_train_norm =scaled_x_data[:int(X.shape[0] * 0.8)]
Y_train_norm =scaled_y_data[:int(X.shape[0] * 0.8)]
X_val_norm = scaled_x_data[int(X.shape[0] * 0.8):]
Y_val_norm = scaled_y_data[int(X.shape[0] * 0.8):]

#构建BP神经网络
model = Sequential()
model.add(Dense(20,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam',metrics=['mse'])
model.summary()
#训练
history = model.fit(X_train_norm, Y_train_norm, epochs=50,batch_size=32, validation_data=(X_val_norm, Y_val_norm))
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(range(len(loss)), loss, 'b-', label='train_loss')
plt.plot(range(len(loss)), val_loss, 'r-', label='test_loss')
plt.legend(loc='best')
plt.show()

model_pred = model.predict(X_val_norm)
#返归一化
val_pred=scaler_for_y.inverse_transform(model_pred.reshape((-1,1)))
# 实际值与预测值对比图
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_val,val_pred)
rmse=np.sqrt(mse)
acc=1-np.mean(np.abs((Y_val[:, 0]-val_pred[:, 0])/val_pred[:, 0]))




