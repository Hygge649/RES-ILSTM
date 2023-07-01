import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.layers import *
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, LambdaCallback
from sklearn.metrics import roc_auc_score
import math
import matplotlib.pyplot as plt
import os


def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)

def build_modle(look_back, dim):
    # 构建 LSTM 网络
    model = Sequential()
    model.add(LSTM(32, input_length=50, input_dim=dim, activation='tanh'))
    # model.add(LSTM(32, input_length=50, input_dim=dim, activation='selu', kernel_initializer='lecun_normal'))
    model.add(Dense(dim))  # dense(activation =None, d(x)=x
    return model


# from keras.regularizers import l2
#
# def build_modle(look_back, dim):
#     # 设置 L2 正则化系数
#     reg_coeff = 0.01
#
#     # 构建 LSTM 网络
#     #l2 正则化函数的默认参数 lambda（即正则化系数）是 0.01
#     model = Sequential()
#     model.add(LSTM(32, input_length=50, input_dim=dim, activation='tanh', kernel_regularizer=l2(reg_coeff)))
#     # model.add(LSTM(32, input_length=50, input_dim=dim, activation='selu', kernel_initializer='lecun_normal'))
#     model.add(Dense(dim, kernel_regularizer=l2(reg_coeff)))  # dense(activation=None, d(x)=x)
#     return model

def data_depart(file):
    train_size = int(len(file) * 0.8)
    test_size = int(len(file)*0.2)
    file_train = file.iloc[:train_size, ]
    # file_valid = file.iloc[train_size:train_size + test_size, ]
    file_test = file.iloc[train_size:, ]
    return file_train, file_test

#('Mean_Speed','V_DIFF', 'GAP') delta(v,v_diff,gap)

#mse,rmse
def mse_rmse(records_real,records_pre):
    if len(records_real) == len(records_pre):
        mse = sum([(x - y) ** 2 for x, y in zip(records_real, records_pre)]) / len(records_real)
        rmse= math.sqrt(mse)
        return mse, rmse
    else:
        return None


#load data: data_truth, data_diff, data_idm
filefoldnames = r'G:\train'
count = len(os.listdir(filefoldnames))
arr = np.zeros((1, 2))
diff_idm_v_loss = pd.DataFrame(arr, columns=['MSE_V', 'RMSE_V'], index=['train'])
diff_idm_gap_loss = pd.DataFrame(arr, columns=['MSE_GAP', 'RMSE_GAP'], index=['train'])

arr1 = np.zeros((1, 4))
diff_idm_loss = pd.DataFrame(arr1,
                             columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'],
                             index=['train'])

model_performance = pd.DataFrame(arr, columns=['LOSS', 'ACCURACY'], index=['train'])

test_loss = []

for filefoldname in os.listdir(filefoldnames):
    filefold = filefoldnames + '\\' + filefoldname
    file = os.listdir(filefold)

    cur_test_loss = []
    cur_test_loss.append(filefoldname)

    # truth
    truth_path = os.path.join(filefold, file[4])
    data_truth = pd.read_csv(truth_path, delim_whitespace=True, encoding='utf-8')
    data_truth = data_truth[["Frame_ID", "Mean_Speed", "gap"]]
    data_train, data_test = data_depart(data_truth)

    #data_idm
    path_idm = os.path.join(filefold, file[1])
    data_idm = pd.read_csv(path_idm, delim_whitespace=True, encoding='utf-8')
    data_idm = data_idm[["Mean_Speed", "Speed_Diff", "gap"]]

    #data_diff
    path_diff = os.path.join(filefold, file[0])
    data_diff = pd.read_csv(path_diff,  delim_whitespace=True,
                            names=['Frame_ID', 'DIFF_Speed', 'DIFF_Speed_Diff', 'DIFF_gap'],
                            header=0, encoding='utf-8')
    data_diff = data_diff[['Frame_ID', 'DIFF_Speed', 'DIFF_gap']]
    data = pd.concat([data_diff, data_idm], axis=1)
    data = data.iloc[:, 1:]
    dim = 5

    #final output:  lstm_diff + idm
    diff = pd.DataFrame(data=None, columns=['Frame_ID',  'DIFF_Speed', 'DIFF_gap', 'Mean_Speed', 'Speed_Diff', 'gap'])
    # diff = pd.DataFrame(data=None, columns=['Frame_ID',  'DIFF_gap', 'Mean_Speed', 'Speed_Diff', 'gap'])
    diff['Frame_ID'] = data_truth['Frame_ID']


    # 缩放数据
    # 数据归一化可以提升模型收敛速度，加快梯度下降求解速度，提升模型精度，消除量纲得影响，简化计算
    scaler = MinMaxScaler(feature_range=(0, 1))
    #data = scaler.fit_transform(data)  #ndarray

    # train_size = int(0.8*len(data))
    train = scaler.fit_transform(data)



    # data = scaler.inverse_transform(data)

    # 预测数据步长 50->1
    look_back = 50
    trainX, trainY = create_dataset(train, look_back)

    model = build_modle(look_back, dim)
    #default 0.001
    # Adam = keras.optimizers.adam_v2.Adam(learning_rate=0.0009, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer="Adam", metrics=["accuracy"])
    # model.summary()

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
    )
    # batch_print_callback = LambdaCallback(
    #     on_epoch_end=lambda batch, logs: print(model.evaluate(testX, testY)))
    # metric = []
    # batch_print_callback = LambdaCallback(
    #     on_epoch_end=lambda batch, logs: metric.append(model.evaluate(testX, testY)))
    history = model.fit(trainX, trainY,
                        batch_size=32, epochs=200,
                        # validation_data=(validX, validY),
                        callbacks=[early_stopping],#, batch_print_callback],
                        verbose=2, shuffle=False)


    model.save(filepath='../DIFF_model/diff_v_gap_model.h5', overwrite=True, include_optimizer=True)

    #for the whole training process
    # metric = np.array(metric)
    plt.plot(history.history['loss'], label='train_loss')
    # plt.plot(history.history['val_loss'], label='val_loss')
    # plt.plot(metric[:, 0], label='test_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='train_acc')
    # plt.plot(history.history['val_accuracy'], label='val_acc')
    # plt.plot(metric[:, 1], label='test_acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()

    # for the best epoch
    best_epoch = early_stopping.stopped_epoch
    best_train_loss = history.history['loss'][best_epoch]
    best_train_acc = history.history['accuracy'][best_epoch]
    # best_val_loss = history.history['val_loss'][best_epoch]
    # best_val_acc = history.history['val_accuracy'][best_epoch]
    # best_test_loss, best_test_acc = model.evaluate(testX, testY)

    # cur_test_loss.append(best_test_loss)
    test_loss.append(cur_test_loss)

    best_epoch_performance = pd.DataFrame(
        data=[[best_train_loss, best_train_acc]],
              # [best_val_loss, best_val_acc],
              # [best_test_loss, best_test_acc]],
        columns=['LOSS', 'ACCURACY'],
        index=['train'])

    model_performance += best_epoch_performance

    # 对训练数据的Y进行预测
    trainPredict = model.predict(trainX)
    # 对验证数据的Y进行预测
    # validPredict = model.predict(validX)
    # 对测试数据的Y进行预测
    # testPredict = model.predict(testX)

    # 对数据进行逆缩放
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    # validPredict = scaler.inverse_transform(validPredict)
    # validY = scaler.inverse_transform(validY)
    # testPredict = scaler.inverse_transform(testPredict)
    # testY = scaler.inverse_transform(testY)

    diff.iloc[look_back:len(trainPredict)+look_back, 1:] = trainPredict


    #lstm fo diff is diff(df)
    #and then plus idm output is lstm_data
    #data_sim3
    LSTM_data = pd.DataFrame(data=None, columns=['Frame_ID', 'Mean_Speed', 'gap'])
    LSTM_data['Frame_ID'] = diff['Frame_ID']
    LSTM_data['Mean_Speed'] = diff['DIFF_Speed'] + data_idm['Mean_Speed']
    LSTM_data['gap'] = diff['DIFF_gap'] + data_idm['gap']

    # LSTM_path = filefold + '/' + 'DIFF_IDM.txt'
    # LSTM_path = filefold + '/' + 'DIFF_IDM_1.txt'
    # LSTM_path = filefold + '/' + 'DIFF_IDM_2.txt'
    # LSTM_path = filefold + '/' + 'DIFF_IDM_3.txt'
    # LSTM_data.to_csv(LSTM_path, sep='\t', index=False)


    # train
    train_mse_v, train_rmse_v = mse_rmse(trainPredict[look_back:, 1], trainY[look_back:, 1])
    train_mse_gap, train_rmse_gap = mse_rmse(trainPredict[look_back:, -1], trainY[look_back:, -1])
    # # valid
    # valid_mse_v, valid_rmse_v = mse_rmse(data_valid.iloc[look_back:, 1], sim3_valid.iloc[look_back:, 1])
    # valid_mse_gap, valid_rmse_gap = mse_rmse(data_valid.iloc[look_back:, -1], sim3_valid.iloc[look_back:, -1])
    # test
    # test_mse_v, test_rmse_v = mse_rmse(data_test.iloc[look_back:, 1], sim3_test.iloc[look_back:, 1])
    # test_mse_gap,  test_rmse_gap = mse_rmse(data_test.iloc[look_back:, -1], sim3_test.iloc[look_back:, -1])


    LSTM_DIFF_V_ERROR = pd.DataFrame(
        data=[[train_mse_v, train_rmse_v]],
        columns=['MSE_V', 'RMSE_V'],
        index=['train'])


    LSTM_DIFF_gap_ERROR = pd.DataFrame(
        data=[[train_mse_gap, train_rmse_gap]],
        columns=['MSE_GAP', 'RMSE_GAP'],
        index=['train'])

    LSTM_DIFF_ERROR = pd.DataFrame(
        data=[[train_mse_v, train_mse_gap, train_rmse_v, train_rmse_gap]],
        columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'],
        index=['train'])




    # # LSTM_DIFF_ERROR_path = filefold + '/' + 'DIFF_IDM_ERROR.txt'
    # # LSTM_DIFF_ERROR_path = filefold + '/' + 'DIFF_IDM_ERROR_1.txt'
    # # LSTM_DIFF_ERROR_path = filefold + '/' + 'DIFF_IDM_ERROR_2.txt'
    # LSTM_DIFF_ERROR_path = filefold + '/' + 'DIFF_IDM_ERROR_3.txt'
    #
    # LSTM_DIFF_ERROR.to_csv(LSTM_DIFF_ERROR_path, sep='\t')


    # diff_idm_v_loss += LSTM_DIFF_V_ERROR
    # diff_idm_gap_loss += LSTM_DIFF_gap_ERROR
    diff_idm_loss += LSTM_DIFF_ERROR

# test_loss = pd.DataFrame(test_loss)

model_performance = model_performance/count
print(model_performance)
# diff_idm_loss = diff_idm_v_loss / count
# diff_idm_loss = diff_idm_gap_loss / count
diff_idm_loss = diff_idm_loss / count
print(diff_idm_loss)



'''



'''
'''
0.IDM:
           MSE_V     MSE_GAP    RMSE_V  RMSE_GAP
train   0.037265    0.660449  0.133469  0.768096
test   34.210751  653.439062  1.510930  6.966411

0.IDM:(delete ...
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.042585  0.660598  0.137987  0.758490
test   0.131710  2.810674  0.208769  1.280959

1.LSTM:
lr=0.0008:LSTM_model.h5
           LOSS  ACCURACY
train  0.009199  0.915750
test   0.005380  0.928651
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.067921  0.826906  0.254287  0.737561
test   0.314369  2.135231  0.425677  0.877262




1.(IDM,+ DIFF_V):  lr=0.0009 is better

            LOSS  ACCURACY
train   0.009889  0.870548
valid   0.080515  0.839715
test   80.925350  0.862494
           MSE_V    RMSE_V
train   0.004162  0.046207
valid   0.005590  0.057809
test   53.550821  1.67422

delete and depart:
lr=0.0010:
           LOSS  ACCURACY
train  0.003011  0.895851
test   0.246648  0.870393
          MSE_V    RMSE_V
train  0.002575  0.037471
test   0.112685  0.119694       

lr=0.0009:
           LOSS  ACCURACY
train  0.003089  0.893341
test   0.169461  0.904190
          MSE_V    RMSE_V
train  0.002473  0.038036
test   0.074377  0.110292

lr=0.0008:
           LOSS  ACCURACY
train  0.003273  0.893329
test   0.171952  0.900202
          MSE_V    RMSE_V
train  0.002255  0.037589
test   0.080164  0.113998


2.(IDM,+ DIFF_gap):  lr=0.0010 is better

             LOSS  ACCURACY
train    0.016720  0.866053
valid    0.045961  0.853374
test   111.549055  0.801271
           MSE_GAP  RMSE_GAP
train     0.045689  0.143707
valid     0.059070  0.185071
test   1072.666637  8.111194

delete and depart:
           LOSS  ACCURACY
train  0.002692  0.894847
test   0.156967  0.893678
        MSE_GAP  RMSE_GAP
train  0.041048  0.132773
test   1.551005  0.666487

lr=0.0009:
           LOSS  ACCURACY
train  0.002836  0.895663
test   0.164150  0.915481
        MSE_GAP  RMSE_GAP
train  0.033358  0.132186
test   1.779664  0.713230



    
3.(IDM, + DIFF_Speed, DIFF_gap): lr=0.001 is better 
lr=0.001:
             LOSS  ACCURACY
train    0.007937  0.848571
valid    0.072599  0.749108
test   149.423848  0.790512
           MSE_V      MSE_GAP    RMSE_V  RMSE_GAP
train   0.003290     0.033985  0.042926  0.136508
valid   0.006229     0.063874  0.059080  0.195387
test   54.483423  1082.505659  1.761356  8.320726

           LOSS  ACCURACY
train  0.003092  0.882196
test   0.301865  0.905930
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.002340  0.025678  0.037411  0.120984
test   0.103524  1.616504  0.134133  0.709560

lr=0.0009:
           LOSS  ACCURACY
train  0.003471  0.876033
test   0.329812  0.819957
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.002599  0.028195  0.038284  0.125479
test   0.119046  1.886869  0.148318  0.760418




final result:
IDM+V/GAP is better than IDM_V_GAP
 








lr=0.0009:
             LOSS  ACCURACY
train    0.004130  0.879514
valid    0.044359  0.828360
test   146.913097  0.729456
           MSE_V      MSE_GAP    RMSE_V  RMSE_GAP
train   0.004349     0.046902  0.045633  0.145516
valid   0.005897     0.050815  0.060467  0.174260
test   54.066341  1055.977687  1.754247  8.198416


lr=0.009, earlystopping_ loss:
             LOSS  ACCURACY
train    0.003982  0.870877
valid    0.042575  0.787617
test   146.702369  0.794966
           MSE_V      MSE_GAP    RMSE_V  RMSE_GAP
train   0.003065     0.027084  0.037557  0.114380
valid   0.008709     0.058710  0.067311  0.186733
test   52.974584  1068.335489  1.737713  8.158991

lr=0.009, earlystopping_ acc:
             LOSS  ACCURACY
train    0.005407  0.845006
valid    0.053358  0.769285
test   147.681461  0.682448
           MSE_V      MSE_GAP    RMSE_V  RMSE_GAP
train   0.021592     0.136352  0.067936  0.197768
valid   0.019948     0.154192  0.096746  0.285764
test   53.404700  1072.157869  1.764324  8.369994

l2:
             LOSS  ACCURACY
train    0.027582  0.783939
valid    0.105131  0.675253
test   148.558849  0.672690
           MSE_V      MSE_GAP    RMSE_V  RMSE_GAP
train   0.007865     0.039272  0.060794  0.168351
valid   0.010707     0.090076  0.083711  0.240026
test   54.167602  1073.422432  1.797374  8.532385






'''

    # #draw
    # x = LSTM_data['Frame_ID']
    # y1, y2 = LSTM_data['Mean_Speed'], LSTM_data['gap']  # sim v,gap, df
    # z1, z2 = data_truth['Mean_Speed'], data_truth['gap']
    #
    # plt.title("LSTM_IDM_V")
    # plt.plot(x, y1, c='r', label='LSTM')
    # plt.plot(x, z1, c='g', label='Data')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/DIFF_V.jpg')
    # plt.show()
    #
    # plt.title("LSTM_IDM_GAP")
    # plt.plot(x, y2, c='r', label='LSTM')
    # plt.plot(x, z2, c='g', label='Data')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/DIFF_GAP.jpg')
    # plt.show()
