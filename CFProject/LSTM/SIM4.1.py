import keras
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

"""
INPUT:IDM, DIFF
OUTPUT: PRE_(IDM+DIFF)
"""

#load_data
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

def reshape_dataset(dataset,look_back):
    # 缩放数据
    # 数据归一化可以提升模型收敛速度，加快梯度下降求解速度，提升模型精度，消除量纲得影响，简化计算
    data = dataset.iloc[:, 1:]  # (len,3): V,GAP
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)  # ndarray
    train_size = int(0.7 * len(data))
    train = data[:train_size, :]
    test = data[train_size:, :]


    data = scaler.inverse_transform(data)

    # 预测数据步长 50->1
    look_back = 50
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)

    return trainX, trainY, testX, testY

# load data: data_truth, data_diff, data_idm
filefoldnames = os.listdir(r'C:\Users\SimCCAD\Desktop\TEST500')  # 168


for filefoldname in filefoldnames:
    filefold = 'C:\\Users\\SimCCAD\\Desktop\\TEST500' + '\\' + filefoldname
    # filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
    file = os.listdir(filefold)

    #truth
    truth_path = filefold + '\\' + file[16]
    data = pd.read_csv(truth_path, delim_whitespace=True, encoding='utf-8')
    data = data[["Frame_ID", "Mean_Speed", "gap"]]

    IDM_path = filefold + '\\' + file[4]
    idm_data = pd.read_csv(IDM_path, delim_whitespace=True, encoding='utf-8')

    #only diff
    DIFF_path = filefold + '\\' + file[0]
    diff_data = pd.read_csv(DIFF_path, delim_whitespace=True, encoding='utf-8')
    diff_data = diff_data[["Frame_ID", "Mean_Speed", "gap"]]


    PRE_data = pd.DataFrame(data=None, columns=['Frame_ID', 'Mean_Speed', 'gap'])
    PRE_data['Frame_ID'] = idm_data['Frame_ID']

    look_back = 50

    trainx1, trainy1, testx1, testy1 = reshape_dataset(idm_data, look_back)
    trainx2, trainy2, testx2, testy2 = reshape_dataset(diff_data, look_back)
    trainx3, trainy3, testx3, testy3 = reshape_dataset(data, look_back)


    #build_model
    input1 = keras.Input(shape=(50, 2), name="IDM")
    input2 = keras.Input(shape=(50, 2), name="DIFF")

    lstm1 = LSTM(32, input_length=50, input_dim=2, activation='tanh')(input1)
    out1 = Dense(2, name="out1")(lstm1)

    lstm2 = LSTM(32, input_length=50, input_dim=2, activation='tanh')(input2)
    out2 = Dense(2, name="out2")(lstm2)
    out = Add()([0.000001*out1, out2])
    model = Model(inputs=[input1, input2], outputs=out)
    # # model.summary()

    model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["accuracy"])
    early_stopping = EarlyStopping(patience=10)
    # model.fit({"IDM": trainx1, "DIFF": trainx2}, {"out1" : trainy1, "out2":trainy2}, epochs=50)
    # model.fit({"IDM": trainx1, "DIFF": trainx2}, trainy2, epochs=50)
    history = model.fit({"IDM": trainx1, "DIFF": trainx2}, trainy2, epochs=200,
                        validation_data=({"IDM": testx1, "DIFF": testx2}, testy2), callbacks=[early_stopping], verbose=2,shuffle=False)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='test_acc')
    plt.title('model loss and acc')
    plt.ylabel('loss and acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    # 对训练数据的Y进行预测
    trainPredict = model.predict([trainx1, trainx2])
    # 对测试数据的Y进行预测
    testPredict = model.predict([testx1, testx2])

    #inverse

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.iloc[:,1:])
    data = scaler.inverse_transform(data) #555,2

    # 对数据进行逆缩放
    trainPredict = scaler.inverse_transform(trainPredict) #478,2
    trainY = scaler.inverse_transform(trainy3)
    testPredict = scaler.inverse_transform(testPredict) #177,2
    testY =scaler.inverse_transform(testy3)

    PRE_data.iloc[look_back:len(trainPredict) + look_back, 1:] = trainPredict
    PRE_data.iloc[len(trainPredict) + 2 * look_back:len(data), 1:] = testPredict

    hybrid_data = pd.DataFrame(data=None, columns=['Frame_ID', 'Mean_Speed', 'gap'])
    hybrid_data['Frame_ID'] = PRE_data['Frame_ID']

    hybrid_data['Mean_Speed'] = PRE_data['Mean_Speed'] + idm_data['Mean_Speed']
    hybrid_data['gap'] = PRE_data['gap'] + idm_data['gap']


    hybrid_path = filefold + '/(DIFF_IDM)_DIFF.txt'
    # hybrid_data.to_csv(hybrid_path, sep='\t', index=False)

    # draw
    x = hybrid_data['Frame_ID']
    y1, y2 = hybrid_data['Mean_Speed'], hybrid_data['gap']  # sim v,gap, df
    z1, z2 = data[:,0], data[:,1]

    plt.title("LSTM_(DIFF_IDM)_DIFF_V")
    plt.plot(x, y1, c='r', label='LSTM_(DIFF_IDM)_DIFF')
    plt.plot(x, z1, c='g', label='Data')
    plt.legend()
    plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/' + 'LSTM_(DIFF_IDM)_DIFF_V.jpg')
    plt.show()

    plt.title("LSTM_(DIFF_IDM)_DIFF_GAP")
    plt.plot(x, y2, c='r', label='LSTM_(DIFF_IDM)_DIFF')
    plt.plot(x, z2, c='g', label='Data')
    plt.legend()
    plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/' + 'LSTM_(DIFF_IDM)_DIFF_GAP.jpg')
    plt.show()




