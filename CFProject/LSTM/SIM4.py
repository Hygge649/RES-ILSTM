import math
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping


"""
INPUT:IDM, DIFF
OUTPUT: PRE_(IDM+DIFF)
"""


# load_data
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


def reshape_dataset(dataset, look_back):
    data = dataset.iloc[:, 1:]  # (len,3): V,V_DIFF,GAP
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)  # ndarray
    train_size = int(0.6 * len(data))
    test_size = int(0.2 * len(data))
    train = data[:train_size, :]
    valid = data[train_size:train_size + test_size, :]
    test = data[train_size + test_size:, :]

    data = scaler.inverse_transform(data)

    # 预测数据步长 50->1
    look_back = 50
    trainX, trainY = create_dataset(train, look_back)
    validX, validY = create_dataset(valid, look_back)
    testX, testY = create_dataset(test, look_back)

    return trainX, trainY, validX, validY, testX, testY


def data_depart(file):
    train_size = int(len(file) * 0.6)
    test_size = int(len(file) * 0.2)
    file_train = file.iloc[:train_size, ]
    file_valid = file.iloc[train_size:train_size + test_size, ]
    file_test = file.iloc[train_size + test_size:, ]
    return file_train, file_valid, file_test


# mse,rmse
def mse_rmse(records_real, records_pre):
    if len(records_real) == len(records_pre):
        mse = sum([(x - y) ** 2 for x, y in zip(records_real, records_pre)]) / len(records_real)
        rmse = math.sqrt(mse)
        return mse, rmse
    else:
        return None


# load data: data_truth, data_diff, data_idm
filefoldnames = os.listdir(r'C:\Users\SimCCAD\Desktop\TEST')

for filefoldname in filefoldnames:
    filefold = 'C:\\Users\\SimCCAD\\Desktop\\TEST' + '\\' + filefoldname
    # filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
    file = os.listdir(filefold)

    # truth
    truth_path = filefold + '\\' + file[9]
    data = pd.read_csv(truth_path, delim_whitespace=True, encoding='utf-8')
    data = data[["Frame_ID", "Mean_Acceleration", "Mean_Speed", "LocalY", "gap"]]
    data_train, data_valid, data_test = data_depart(data)

    IDM_path = filefold + '\\' + file[3]
    idm_data = pd.read_csv(IDM_path, delim_whitespace=True, encoding='utf-8')

    # only diff,ACC	Mean_Speed	LOC	gap
    DIFF_path = filefold + '\\' + file[0]
    diff_data = pd.read_csv(DIFF_path, delim_whitespace=True, encoding='utf-8')
    # diff_data = diff_data[["Frame_ID", "Mean_Speed", "gap"]]

    PRE_data = pd.DataFrame(data=None, columns=['Frame_ID', 'Mean_Speed', 'gap'])
    PRE_data['Frame_ID'] = idm_data['Frame_ID']

    look_back = 50

    trainx1, trainy1, validx1, validy1, testx1, testy1 = reshape_dataset(idm_data, look_back)
    trainx2, trainy2, validx2, validy2, testx2, testy2 = reshape_dataset(diff_data, look_back)
    trainx3, trainy3, validx3, validy3, testx3, testy3 = reshape_dataset(data, look_back)

    # build_model
    input1 = keras.Input(shape=(50, 4), name="IDM")
    input2 = keras.Input(shape=(50, 4), name="DIFF")

    lstm1 = LSTM(32, input_length=50, input_dim=4, activation='tanh')(input1)
    out1 = Dense(4, name="out1")(lstm1)

    lstm2 = LSTM(32, input_length=50, input_dim=4, activation='tanh')(input2)
    out2 = Dense(4, name="out2")(lstm2)
    model = Model(inputs=[input1, input2], outputs=[out1, out2])
    # out = Add()([out1, out2])
    # model = Model(inputs=[input1, input2], outputs=out)
    # model.summary()
    my_metrics = {'out1': "accuracy", 'out2': "accuracy"}

    # model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["accuracy"])
    model.compile(loss={'out1': 'mean_squared_error', 'out2': 'mean_squared_error'},
                  loss_weights={'out1': 1, 'out2': 2},
                  optimizer="Adam", metrics=my_metrics)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
    )
    # model.fit({"IDM": trainx1, "DIFF": trainx2}, {"out1" : trainy1, "out2":trainy2}, epochs=50)
    # model.fit({"IDM": trainx1, "DIFF": trainx2}, trainy2, epochs=50)

    history = model.fit({"IDM": trainx1, "DIFF": trainx2}, {'out1': trainy3, 'out2': trainy1}, epochs=200,
                        validation_data=({"IDM": validx1, "DIFF": validx2}, {'out1': validy3, 'out2': validy1}),
                        callbacks=[early_stopping], verbose=2, shuffle=False)


    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    # plt.plot(history.history['acc'], label='train_acc')
    # plt.plot(history.history['val_acc'], label='val_acc')
    # plt.title('model loss and acc')
    # plt.ylabel('loss and acc')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    # 对训练数据的Y进行预测
    trainPredict = model.predict([trainx1, trainx2])
    # 对验证数据的Y进行预测
    validPredict = model.predict([validx1, validx2])
    # 对测试数据的Y进行预测
    testPredict = model.predict([testx1, testx2])

    # inverse

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.iloc[:, 1:])
    data = scaler.inverse_transform(data)  # 555,2

    # 对数据进行逆缩放
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainy3)
    validPredict = scaler.inverse_transform(validPredict)
    validY = scaler.inverse_transform(validy3)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testy3)

    PRE_data.iloc[look_back:len(trainPredict) + look_back, 1:] = trainPredict
    PRE_data.iloc[len(trainPredict) + 2 * look_back:len(trainPredict) + 2 * look_back + len(validPredict),
    1:] = validPredict
    PRE_data.iloc[
    len(trainPredict) + 3 * look_back + len(validPredict):len(trainPredict) + 3 * look_back + len(validPredict) + len(
        testPredict), 1:] = testPredict

    PRE_path = filefold + '/LSTM_DIFF_IDM.txt'
    PRE_data.to_csv(PRE_path, sep='\t', index=False)

    sim4_train, sim4_valid, sim4_test = data_depart(PRE_data)

    # train
    _, train_rmse_v = mse_rmse(data_train.iloc[look_back:, 1], sim4_train.iloc[look_back:, 1])
    _, train_rmse_gap = mse_rmse(data_train.iloc[look_back:, -1], sim4_train.iloc[look_back:, -1])
    # valid
    _, valid_rmse_v = mse_rmse(data_valid.iloc[look_back:, 1], sim4_valid.iloc[look_back:, 1])
    _, valid_rmse_gap = mse_rmse(data_valid.iloc[look_back:, -1], sim4_valid.iloc[look_back:, -1])
    # test
    _, test_rmse_v = mse_rmse(data_test.iloc[50:, 1], sim4_test.iloc[50:, 1])
    _, test_rmse_gap = mse_rmse(data_test.iloc[50:, -1], sim4_test.iloc[50:, -1])

    LSTM_DIFF_IDM_ERROR = pd.DataFrame(
        data=[[train_rmse_v, train_rmse_gap], [valid_rmse_v, valid_rmse_gap], [test_rmse_v, test_rmse_gap]],
        columns=['RMSE_V', 'RMSE_GAP'],
        index=['train', 'valid', 'test'])

    LSTM_DIFF_IDM_ERROR_path = filefold + '/' + 'LSTM_DIFF_IDM_ERROR.txt'
    LSTM_DIFF_IDM_ERROR.to_csv(LSTM_DIFF_IDM_ERROR_path, sep='\t')

    # # draw
    # x = PRE_data['Frame_ID']
    # y1, y2 = PRE_data['Mean_Speed'], PRE_data['gap']  # sim v,gap, df
    # z1, z2 = data[:,0], data[:,1]
    #
    # plt.title("LSTM_(DIFF_IDM)_V")
    # plt.plot(x, y1, c='r', label='LSTM_(DIFF_IDM)')
    # plt.plot(x, z1, c='g', label='Data')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/' + 'LSTM_(DIFF_IDM)_V.jpg')
    # plt.show()
    #
    # plt.title("LSTM_(DIFF_IDM)_GAP")
    # plt.plot(x, y2, c='r', label='LSTM_(DIFF_IDM)')
    # plt.plot(x, z2, c='g', label='Data')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/' + 'LSTM_(DIFF_IDM)_GAP.jpg')
    # plt.show()
