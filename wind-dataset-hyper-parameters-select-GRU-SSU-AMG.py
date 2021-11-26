# -*- coding: utf-8 -*-
"""
@author: Wendong Zheng
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU

from keras import optimizers
from keras.callbacks import ModelCheckpoint
import numpy as np
import time

np.set_printoptions(threshold=np.inf)#显示np.array的全部数据信息

import sys
# logger txt: python console to x.txt
class Logger(object):
    def __init__(self, filename='./wind_dataset_select_ratio_GRU_SSU_AMG/default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
   
sys.stdout = Logger('./wind_dataset_select_ratio_GRU_SSU_AMG/a.log', sys.stdout)
sys.stderr = Logger('./wind_dataset_select_ratio_GRU_SSU_AMG/a.log_file', sys.stderr)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# load dataset
dataset = read_csv('turbine.csv', header=0, index_col=0)
values = dataset.values

# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[5,6,7]], axis=1, inplace=True)
print(reframed.head())
# split into train and test sets
values = reframed.values
n_train_10minutes = 35371#50530*0.7 is training set.
train = values[:n_train_10minutes, :]
test = values[n_train_10minutes:, :]

#split
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
test_X1, test_y1 = test[:, :-1], test[:, -1]
test_X2, test_y2 = test[:, :-1], test[:, -1]
test_X3, test_y3 = test[:, :-1], test[:, -1]
test_X4, test_y4 = test[:, :-1], test[:, -1]
test_X5, test_y5 = test[:, :-1], test[:, -1]

#reshape
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))
test_X2 = test_X2.reshape((test_X2.shape[0], 1, test_X2.shape[1]))
test_X3 = test_X3.reshape((test_X3.shape[0], 1, test_X3.shape[1]))
test_X4 = test_X4.reshape((test_X4.shape[0], 1, test_X4.shape[1]))
test_X5 = test_X5.reshape((test_X5.shape[0], 1, test_X5.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# design network optimizer=adamg
print('GRU_SSU_AMG select_ratio=0.6 on wind power dataset')
model = Sequential()
model.add(GRU(128, input_shape=(train_X.shape[1], train_X.shape[2]),  implementation=2,select_ratio=0.6))
model.add(Dense(1))
AdaMG = optimizers.AdaMG(lr=0.001)
model.compile(loss='mae', optimizer=AdaMG,metrics=['mae'])
model.summary()
checkpoint = ModelCheckpoint(filepath='./wind_dataset_select_ratio_GRU_SSU_AMG/my_model_best_weights_sr_0_6.h5',save_weights_only=True,monitor='val_mean_absolute_error',mode='auto',save_best_only=True)
# fit network
history = model.fit(train_X, train_y, epochs=30, batch_size=128, validation_data=(test_X, test_y), verbose=2, shuffle=False, callbacks=[checkpoint])


print('GRU_SSU_AMG select_ratio=0.7 on wind power dataset')
model1 = Sequential()
model1.add(GRU(128, input_shape=(train_X.shape[1], train_X.shape[2]),  implementation=2,select_ratio=0.6))#使用GRU的implementation=2新的更新hidden策略来实现
model1.add(Dense(1))
AdaMG = optimizers.AdaMG(lr=0.001)
model1.compile(loss='mae', optimizer=AdaMG,metrics=['mae'])
model1.summary()
checkpoint1 = ModelCheckpoint(filepath='./wind_dataset_select_ratio_GRU_SSU_AMG/my_model_best_weights_sr_0_7.h5',save_weights_only=True,monitor='val_mean_absolute_error',mode='auto',save_best_only=True)
# fit network
history1 = model1.fit(train_X, train_y, epochs=30, batch_size=128, validation_data=(test_X1, test_y1), verbose=2, shuffle=False, callbacks=[checkpoint1])


# design network optimizer=adamg
print('GRU_SSU_AMG select_ratio=0.8 on wind power dataset')
model2 = Sequential()
model2.add(GRU(128, input_shape=(train_X.shape[1], train_X.shape[2]),  implementation=2,select_ratio=0.8))#使用GRU的implementation=2新的更新hidden策略来实现
model2.add(Dense(1))
AdaMG = optimizers.AdaMG(lr=0.001)
model2.compile(loss='mae', optimizer=AdaMG,metrics=['mae'])
model2.summary()
checkpoint2 = ModelCheckpoint(filepath='./wind_dataset_select_ratio_GRU_SSU_AMG/my_model_best_weights_sr_0_8.h5',save_weights_only=True,monitor='val_mean_absolute_error',mode='auto',save_best_only=True)
# fit network
history2 = model2.fit(train_X, train_y, epochs=30, batch_size=128, validation_data=(test_X2, test_y2), verbose=2, shuffle=False, callbacks=[checkpoint2])


# design network optimizer=adamg
print('GRU_SSU_AMG select_ratio=0.9 on wind power dataset')
model3 = Sequential()
model3.add(GRU(128, input_shape=(train_X.shape[1], train_X.shape[2]),  implementation=2,select_ratio=0.8))#使用GRU的implementation=2新的更新hidden策略来实现
model3.add(Dense(1))
AdaMG = optimizers.AdaMG(lr=0.001)
model3.compile(loss='mae', optimizer=AdaMG,metrics=['mae'])
model3.summary()
checkpoint3 = ModelCheckpoint(filepath='./wind_dataset_select_ratio_GRU_SSU_AMG/my_model_best_weights_sr_0_9.h5',save_weights_only=True,monitor='val_mean_absolute_error',mode='auto',save_best_only=True)
# fit network
history3 = model3.fit(train_X, train_y, epochs=30, batch_size=128, validation_data=(test_X3, test_y3), verbose=2, shuffle=False, callbacks=[checkpoint3])


# plot history train-loss
pyplot.ylabel("Train loss value")  
pyplot.xlabel("The number of epochs")  
pyplot.title("Loss function-epoch curves on Wind power dataset")
pyplot.grid()
pyplot.plot(history.history['loss'], label='train_select_ratio=0.6')
pyplot.plot(history1.history['loss'], label='train_select_ratio=0.7')
pyplot.plot(history2.history['loss'], label='train_select_ratio=0.8')
pyplot.plot(history3.history['loss'], label='train_select_ratio=0.9')
pyplot.legend()
pyplot.savefig('./wind_dataset_select_ratio_GRU_SSU_AMG/Figure-wind-train-loss-GRU_SSU_AMG-select_ratio.png', dpi=400)
pyplot.show()


save = history2.history['loss']
file=open('./wind_dataset_select_ratio_GRU_SSU_AMG/GRU_SSU_AMG-train_loss-select_ratio-0-8.txt','w')
file.write(str(save));
file.close()

save1 = history1.history['loss']
file1=open('./wind_dataset_select_ratio_GRU_SSU_AMG/GRU_SSU_AMG-train_loss-select_ratio-0-7.txt','w')
file1.write(str(save1));
file1.close()

save2 = history.history['loss']
file2=open('./wind_dataset_select_ratio_GRU_SSU_AMG/GRU_SSU_AMG-train_loss-select_ratio-0-6.txt','w')
file2.write(str(save2));
file2.close()

save6 = history3.history['loss']
file6=open('./wind_dataset_select_ratio_GRU_SSU_AMG/GRU_SSU_AMG-train_loss-select_ratio-0-9.txt','w')
file6.write(str(save6));
file6.close()


# plot history val-loss
pyplot.ylabel("Validation Loss value")  
pyplot.xlabel("The number of epochs")  
pyplot.title("Loss function-epoch curves on Wind power dataset")
pyplot.grid()
pyplot.plot(history.history['val_loss'], label='val_select_ratio=0.6')
pyplot.plot(history1.history['val_loss'], label='val_select_ratio=0.7')
pyplot.plot(history2.history['val_loss'], label='val_select_ratio=0.8')
pyplot.plot(history3.history['val_loss'], label='val_select_ratio=0.9')
pyplot.legend()
pyplot.savefig('./wind_dataset_select_ratio_GRU_SSU_AMG/Figure-wind-val-loss-GRU_SSU_AMG-select_ratio.png', dpi=400)
pyplot.show()


save3 = history2.history['val_loss']
file3=open('./wind_dataset_select_ratio_GRU_SSU_AMG/GRU_SSU_AMG-val_loss-select_ratio_0_8.txt','w')#当前目录
file3.write(str(save3));
file3.close()

save4 = history.history['val_loss']
file4=open('./wind_dataset_select_ratio_GRU_SSU_AMG/GRU_SSU_AMG-val_loss-select_ratio_0_6.txt','w')#当前目录
file4.write(str(save4));
file4.close()

save5 = history1.history['val_loss']
file5=open('./wind_dataset_select_ratio_GRU_SSU_AMG/GRU_SSU_AMG-val_loss-select_ratio_0_7.txt','w')#当前目录
file5.write(str(save5));
file5.close()

save7 = history3.history['val_loss']
file7=open('./wind_dataset_select_ratio_GRU_SSU_AMG/GRU_SSU_AMG-val_loss-select_ratio_0_9.txt','w')#当前目录
file7.write(str(save7));
file7.close()


model.load_weights('./wind_dataset_select_ratio_GRU_SSU_AMG/my_model_best_weights_sr_0_6.h5')
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


model1.load_weights('./wind_dataset_select_ratio_GRU_SSU_AMG/my_model_best_weights_sr_0_7.h5')
yhat1 = model1.predict(test_X1)
test_X1 = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))

# make a prediction adam-Phased_LSTM
model2.load_weights('./wind_dataset_select_ratio_GRU_SSU_AMG/my_model_best_weights_sr_0_8.h5')
yhat2 = model2.predict(test_X2)
test_X2 = test_X2.reshape((test_X2.shape[0], test_X2.shape[2]))

# make a prediction adam-SkipGRU
model3.load_weights('./wind_dataset_select_ratio_GRU_SSU_AMG/my_model_best_weights_sr_0_9.h5')
yhat3 = model3.predict(test_X3)
test_X3 = test_X3.reshape((test_X3.shape[0], test_X3.shape[2]))

# invert scaling for forecast 
inv_yhat2 = concatenate((yhat2, test_X2[:, 1:]), axis=1)
inv_yhat2 = scaler.inverse_transform(inv_yhat2)
inv_yhat2 = inv_yhat2[:,0]

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for forecast 
inv_yhat1 = concatenate((yhat1, test_X1[:, 1:]), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat1)
inv_yhat1 = inv_yhat1[:,0]

# invert scaling for forecast 
inv_yhat3 = concatenate((yhat3, test_X3[:, 1:]), axis=1)
inv_yhat3 = scaler.inverse_transform(inv_yhat3)
inv_yhat3 = inv_yhat3[:,0]

# invert scaling for actual 
inv_y = scaler.inverse_transform(test_X)
inv_y = inv_y[:,0]

# invert scaling for actual 
inv_y2 = scaler.inverse_transform(test_X2)
inv_y2 = inv_y2[:,0]

# invert scaling for actual 
inv_y1 = scaler.inverse_transform(test_X1)
inv_y1 = inv_y1[:,0]

# invert scaling for actual
inv_y3 = scaler.inverse_transform(test_X3)
inv_y3 = inv_y3[:,0]

# calculate RMSE and MAE 
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
mae = mean_absolute_error(inv_y, inv_yhat)
print('Test RMSE-GRU_SSU_AMG-select_ratio-0.6: %.3f' % rmse)
print('Test MAE-GRU_SSU_AMG-select_ratio-0.6: %.3f' % mae)

# calculate RMSE and MAE of Adam-Phased_LSTM
rmse1 = sqrt(mean_squared_error(inv_y1, inv_yhat1))
mae1 = mean_absolute_error(inv_y1, inv_yhat1)
print('Test RMSE-GRU_SSU_AMG-select_ratio-0.7: %.3f' % rmse1)
print('Test MAE-GRU_SSU_AMG-select_ratio-0.7: %.3f' % mae1)

# calculate RMSE and MAE of Adam-GRU-new_hidden_strategy
rmse2 = sqrt(mean_squared_error(inv_y2, inv_yhat2))
mae2 = mean_absolute_error(inv_y2, inv_yhat2)
print('Test RMSE-GRU_SSU_AMG-select_ratio-0.8: %.3f' % rmse2)
print('Test MAE-GRU_SSU_AMG-select_ratio-0.8: %.3f' % mae2)

# calculate RMSE and MAE of Adam-SkipGRU
rmse3 = sqrt(mean_squared_error(inv_y3, inv_yhat3))
mae3 = mean_absolute_error(inv_y3, inv_yhat3)
print('Test RMSE-GRU_SSU_AMG-select_ratio-0.9: %.3f' % rmse3)
print('Test MAE-GRU_SSU_AMG-select_ratio-0.9: %.3f' % mae3)

#true-pred values --> *.txt
a = inv_y
file12=open('./wind_dataset_select_ratio_GRU_SSU_AMG/true_y.txt','w')
file12.write(str(a));
file12.close()

a1 = inv_yhat
file13=open('./wind_dataset_select_ratio_GRU_SSU_AMG/pred_select_ratio_0_6.txt','w')
file13.write(str(a1));
file13.close()

a2 = inv_yhat1
file14=open('./wind_dataset_select_ratio_GRU_SSU_AMG/pred_select_ratio_0_7.txt','w')
file14.write(str(a2));
file14.close()

a3 = inv_yhat3
file15=open('./wind_dataset_select_ratio_GRU_SSU_AMG/pred_select_ratio_0_8.txt','w')
file15.write(str(a3));
file15.close()

a6 = inv_yhat2
file18=open('./wind_dataset_select_ratio_GRU_SSU_AMG/pred_select_ratio_0_9.txt','w')
file18.write(str(a6));
file18.close()

pyplot.title('The grid search effect of select_ratio on the Wind power dataset (the next 4 hours)')
pyplot.xlabel('Time range (/10 minutes)')
pyplot.ylabel('Power range (KWh)')
pyplot.grid()
pyplot.plot(inv_y[:24],label='true')
pyplot.plot(inv_yhat[:24],'r--',label='pred_select_ratio=0.6')
pyplot.plot(inv_yhat1[:24],'g--',label='pred_select_ratio=0.7')
pyplot.plot(inv_yhat2[:24],label='pred_select_ratio=0.8')
pyplot.plot(inv_yhat3[:24],'b--',label='pred_select_ratio=0.9')
pyplot.legend()
pyplot.savefig('./wind_dataset_select_ratio_GRU_SSU_AMG/Figure-wind-GRU_SSU_AMG_select_ratio.png', dpi=400)
pyplot.show()