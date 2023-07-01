import numpy as np
import pandas as pd
import keras
from keras.layers import *
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, LambdaCallback
import math
import matplotlib.pyplot as plt
import os


# 将数据截取成50个一组的监督学习格式
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

# def build_modle(look_back):  #OUTPUT =3, V AND GAP
#     # 构建 LSTM 网络
#     model = Sequential()
#     # input shape是3维: (Batch_size, Time_step, Input_Sizes),
#     model.add(LSTM(32, batch_input_shape=(None, look_back, 3), activation='tanh'))
#     # model.add(LSTM(32, input_length=50, input_dim=3, activation='selu', kernel_initializer='lecun_normal'))
#     model.add(Dense(3))
#     return model
from keras.regularizers import l2

def build_modle(look_back):
    # 设置 L2 正则化系数
    reg_coeff = 0.01

    # 构建 LSTM 网络
    #l2 正则化函数的默认参数 lambda（即正则化系数）是 0.01
    model = Sequential()
    model.add(LSTM(32, input_length=50, input_dim=3, activation='tanh', kernel_regularizer=l2(reg_coeff)))
    # model.add(LSTM(32, input_length=50, input_dim=dim, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Dense(3, kernel_regularizer=l2(reg_coeff)))  # dense(activation=None, d(x)=x)
    return model


def mse_rmse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差, 均方根误差：是均方误差的算术平方根
    """
    if len(records_real) == len(records_predict):
        mse = sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
        rmse = math.sqrt(mse)

        return mse,rmse
    else:
        return None


def data_depart(file):
    train_size = int(len(file) * 0.8)
    valid_size = int(len(file) * 0.2)
    file_train = file.iloc[:train_size, ]
    # file_valid = file.iloc[train_size:train_size + valid_size, ]
    # file_test = file.iloc[train_size + valid_size:, ]
    file_test = file.iloc[train_size :, ]
    return file_train, file_test #file_valid, file_test


def LSTM_ERROR(sim, truth):
    mse_v, rmse_v = mse_rmse(truth[:, 0], sim[:, 0])
    mse_gap, rmse_gap = mse_rmse(truth[:, -1], sim[:, -1])

    LSTM_ERROR = pd.DataFrame(
        data=[[mse_v, mse_gap, rmse_v, rmse_gap]],
        columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'])

    return LSTM_ERROR

#'Mean_Speed', 'Speed_diff', 'gap'
# # all the file
# filefoldnames = os.listdir(r'C:\Users\SimCCAD\Desktop\TEST')  #39
# count = len(filefoldnames)
# arr1 = np.zeros((3, 4))
# LSTM_loss = pd.DataFrame(arr1, columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'], index=['train', 'valid', 'test'])
arr1 = np.zeros((2, 4))
LSTM_loss = pd.DataFrame(arr1, columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'],
                         index=['train', 'test'])

# arr2 = np.zeros((3, 2))
# model_performance = pd.DataFrame(arr2, columns=['LOSS', 'ACCURACY'], index=['train', 'valid', 'test'])


# for filefoldname in filefoldnames:
#     filefold = 'C:\\Users\\SimCCAD\\Desktop\\TEST' + '\\' + filefoldname
#     # filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
#     file = os.listdir(filefold)
#
#     path = filefold + '\\' + file[2]
#     data = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
#     data = data[["Frame_ID", "Mean_Speed", "speed_diff", "gap"]]
#     # data_train, data_valid, data_test = data_depart(data)
#     data_train,  data_test = data_depart(data)
#     name = os.path.splitext(file[2])[0]

def get_data(path):


    all_cf_datas =pd.DataFrame()

    filefoldnames = path
    for filefoldname in os.listdir(filefoldnames):
        filefold = os.path.join(filefoldnames, filefoldname)
        file = os.listdir(filefold)
        path = os.path.join(filefold, file[0])
        train = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')

        all_cf_datas = pd.concat([all_cf_datas, train], axis=0)

    return all_cf_datas


train = get_data(r'G:\train')
train = train[['Frame_ID', 'Mean_Speed', 'speed_diff', 'gap']]
count_train = len(train)

test = get_data(r'G:\test')
test = test[['Frame_ID', 'Mean_Speed', 'speed_diff', 'gap']]
count_test = len(test)

arr2 = np.zeros((2, 2))
model_performance = pd.DataFrame(arr2, columns=['LOSS', 'ACCURACY'], index=['train', 'test'])
test_loss = []

LSTM_train = pd.DataFrame(data=None, columns=['Frame_ID', 'Mean_Speed', 'Speed_diff', 'gap'])
LSTM_train['Frame_ID'] = train['Frame_ID']
LSTM_test = pd.DataFrame(data=None, columns=['Frame_ID', 'Mean_Speed', 'Speed_diff', 'gap'])
LSTM_test['Frame_ID'] = test['Frame_ID']

# 缩放数据
# 数据归一化可以提升模型收敛速度，加快梯度下降求解速度，提升模型精度，消除量纲得影响，简化计算
# data = data.iloc[:,1:]  #(len,3): V,V_DIFF,GAP
scaler = MinMaxScaler(feature_range=(0, 1))

# train, valid, test = data_depart(data)
# train, test = data_depart(data)
train = scaler.fit_transform(train.iloc[:, 1:])
# valid = scaler.transform(valid)
test = scaler.transform(test.iloc[:, 1:])


# 预测数据步长 50->1
look_back = 50
trainX, trainY = create_dataset(train, look_back)
# validX, validY = create_dataset(valid, look_back)
testX, testY = create_dataset(test, look_back)


#input_shape
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 3))
# validX = np.reshape(validX, (validX.shape[0], validX.shape[1], 3))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 3))
#
# model = build_modle(look_back)
# Adam = keras.optimizers.adam_v2.Adam(learning_rate=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
# model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["accuracy"])
# #model.fit(trainX, trainY, epochs=50, verbose=2)
# # model.summary()
#
# early_stopping = EarlyStopping(
#     # monitor='val_loss',
#     monitor='loss',
#     patience=10,
#     restore_best_weights=True,
# )
# metric = []
# batch_print_callback = LambdaCallback(
#     on_epoch_end=lambda batch, logs: metric.append(model.evaluate(testX, testY)))
# history = model.fit(trainX, trainY, epochs=200,
#                     # validation_data=(validX, validY),
#                     callbacks=[early_stopping, batch_print_callback],
#                     verbose=2, shuffle=False)
# model.save(filepath='./data_1min/LSTM_model_l2.h5', overwrite=True, include_optimizer=True)
#
# metric = np.array(metric)
#
#
# #during the whole training process
# plt.plot(history.history['loss'], label='train')
# # plt.plot(history.history['val_loss'], label='val')
# plt.plot(metric[:, 0], label='test')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend()
# # plt.savefig(filefold + '/' + 'loss.jpg')
# plt.show()
#
# plt.plot(history.history['accuracy'], label='train_acc')
# # plt.plot(history.history['val_accuracy'], label='val_acc')
# plt.plot(metric[:, 1], label='test_acc')
# plt.ylabel('acc')
# plt.xlabel('epoch')
# plt.legend()
# # plt.savefig(filefold + '/' + 'acc.jpg')
# plt.show()
#
# #for the best epoch
# best_epoch = early_stopping.stopped_epoch
# best_train_loss = history.history['loss'][best_epoch]
# best_train_acc = history.history['accuracy'][best_epoch]
# # best_val_loss = history.history['val_loss'][best_epoch]
# # best_val_acc = history.history['val_accuracy'][best_epoch]
# best_test_loss, best_test_acc = model.evaluate(testX, testY)
#
# # cur_test_loss.append(best_test_loss)
# # test_loss.append(cur_test_loss)
# test_loss.append(best_test_loss)
#
# best_epoch_performance = pd.DataFrame(
#     data=[[best_train_loss, best_train_acc],
#           # [best_val_loss, best_val_acc],
#           [best_test_loss, best_test_acc]],
#     columns=['LOSS', 'ACCURACY'],
#     index=['train', 'test'])
#     # index=['train', 'valid', 'test'])
#
# model_performance += best_epoch_performance

from keras.models import load_model
model = load_model('../data_1min/LSTM_model_9.h5')

# 对训练数据的Y进行预测
trainPredict = model.predict(trainX)
#对验证数据的Y进行预测
# validPredict = model.predict(validX)
# 对测试数据的Y进行预测
testPredict = model.predict(testX)
# 对数据进行逆缩放
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
# validPredict = scaler.inverse_transform(validPredict)
# validY = scaler.inverse_transform(validY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

LSTM_train.iloc[look_back:, 1:] = trainPredict
LSTM_test.iloc[look_back:, 1:] = testPredict
# LSTM_data.iloc[len(trainPredict)+2*look_back:len(trainPredict)+2*look_back + len(validPredict), 1:] = validPredict
# LSTM_data.iloc[len(trainPredict)+len(validPredict)+3*look_back:len(trainPredict)+3*look_back + len(validPredict)+len(testPredict), 1:] = testPredict

#LSTM_path = r'C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0\ngsim11.0_LSTM.txt'
# LSTM_path = filefold + '/LSTM.txt'
# LSTM_data.to_csv(LSTM_path, sep='\t', index=False)

# sim2_train, sim2_valid, sim2_test = data_depart(LSTM_data)
# sim2_train, sim2_test = data_depart(LSTM_data)



# model_performance = model_performance/count_train
print(model_performance)

train_loss = LSTM_ERROR(trainPredict, trainY)
test_loss = LSTM_ERROR(testPredict, testY)
loss = pd.concat([train_loss, test_loss])
loss.index = ['train', 'test']
print(train_loss)
print(test_loss)


'''
for all trajectory:
0.IDM:
         MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.09778  1.920565  0.152173  0.849595

         MSE_V    MSE_GAP    RMSE_V  RMSE_GAP
test  1.542872  57.225518  0.718114  3.034383


1.LSTM 0.0009 is the best one
lr=0.001:
           LOSS  ACCURACY
train  0.000059  0.995869
test   0.000061  0.993544
        MSE_V   MSE_GAP  RMSE_V  RMSE_GAP
train  0.025249  0.229051  0.1589  0.478592
test   0.022792  0.211771  0.150969  0.460185


lr=0.0009
           LOSS  ACCURACY
train  0.000060  0.995414
test   0.000059  0.994599
      MSE_V  MSE_GAP    RMSE_V  RMSE_GAP
train  0.022026  0.24079  0.148411  0.490704
test   0.020146  0.222803  0.141935   0.47202


lr=0.0008:
           LOSS  ACCURACY
train  0.000058  0.996131
test   0.000058  0.994247
       MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.023843  0.237202  0.154412  0.487034
test   0.021415  0.218725  0.14634  0.467681

lr = 0.0009 +l2:

           LOSS  ACCURACY
train  0.005105  0.935475
test   0.004301  0.949343
       MSE_V    MSE_GAP    RMSE_V  RMSE_GAP
train  0.306485  22.926163  0.553611  4.788127
test   0.26494  18.702321  0.514723  4.324618




for one pair:
0.IDM:
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.042585  0.660598  0.137987  0.758490
test   0.131710  2.810674  0.208769  1.280959

LSTM:
lr=0.0009:
           LOSS  ACCURACY
train  0.002008  0.934519
test   0.005522  0.924683
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.071944  0.880602  0.261162  0.752042
test   0.316897  2.705557  0.462017  0.947240

训练过程中， acc 震荡过大， 调小学习率：

lr=0.0008:LSTM_model.h5
           LOSS  ACCURACY
train  0.009199  0.915750
test   0.005380  0.928651
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.067921  0.826906  0.254287  0.737561
test   0.314369  2.135231  0.425677  0.877262

lr=0.0007:
           LOSS  ACCURACY
train  0.007151  0.922865
test   0.005013  0.929190
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.083031  0.886538  0.279955  0.757396
test   0.331024  2.589184  0.482873  0.935556

'''

    # x= LSTM_data['Frame_ID']
    # y1,y2= LSTM_data['Mean_Speed'],LSTM_data['gap'] #sim v,gap, df
    # z1,z2 = data[:,0],data[:,2]#data, ndarray
    #
    # plt.title("LSTM_V")
    # plt.plot(x,y1,c= 'r',label = 'LSTM' )
    # plt.plot(x,z1,c='g', label='Data')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/LSTM_V.jpg')
    # plt.show()
    #
    # plt.title("LSTM_GAP")
    # plt.plot(x,y2,c= 'r',label = 'LSTM' )
    # plt.plot(x,z2,c='g', label='Data')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/LSTM_GAP.jpg')
    # plt.show()
