
'''
dim = 3:
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 32)                4608

 dense (Dense)               (None, 3)                 99

=================================================================
Total params: 4,707
Trainable params: 4,707
Non-trainable params: 0

    +------------------------------------+
    |                                    |
    |          Input (None, look_back, 3)          |
    |                                    |
    +------------------------------------+
                    |
                    |
                    V
    +------------------------------------+
    |                                    |
    |       LSTM (32 neurons, tanh)      |
    |                                    |
    +------------------------------------+
                    |
                    |
                    V
    +------------------------------------+
    |                                    |
    |           Dense (3 neurons)         |
    |                                    |
    +------------------------------------+
                    |
                    |
                    V
             Output (None, 3)


dim = 4:
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 32)                4736

 dense (Dense)               (None, 4)                 132

=================================================================
Total params: 4,868
Trainable params: 4,868
Non-trainable params: 0


dim = 5
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 lstm (LSTM)                 (None, 32)                4864

 dense (Dense)               (None, 5)                 165

=================================================================
Total params: 5,029
Trainable params: 5,029
Non-trainable params: 0
'''


#final result
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

2.DIFF_LSTM

2.1.(IDM,+ DIFF_V):  lr=0.0009 is better

            LOSS  ACCURACY
train   0.009889  0.870548
valid   0.080515  0.839715
test   80.925350  0.862494
           MSE_V    RMSE_V
train   0.004162  0.046207
valid   0.005590  0.057809
test   53.550821  1.67422

delete and depart:

lr=0.0009:
           LOSS  ACCURACY
train  0.003089  0.893341
test   0.169461  0.904190
          MSE_V    RMSE_V
train  0.002473  0.038036
test   0.074377  0.110292


2.2. (IDM,+ DIFF_gap):  lr=0.0010 is better

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

    
2.3.(IDM, + DIFF_Speed, DIFF_gap): lr=0.001 is better 
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



final result:
IDM+V/GAP is better than IDM_V_GAP
 


'''

# #all the file
# filefoldnames = os.listdir(r'C:\Users\SimCCAD\Desktop\\RENGSIM_600')
# count = len(filefoldnames)
# arr = np.zeros((2, 4))
# idm_loss = pd.DataFrame(arr, columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'], index=['train', 'test'])
# parameter = pd.DataFrame(columns=['Vehicle_ID', 'Maximum_Acc', 'Comfortable_Dec', 'Desire_Spe',
#                                   'Desire_Spa_Tim', 'Minimum_Spa', 'loss(RMSE_ACC)'])

# for filefoldname in filefoldnames: #222
#     filefold = 'C:\\Users\\SimCCAD\\Desktop\\TEST' + '\\' + filefoldname
#     #filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
#     file = os.listdir(filefold)
#     path = filefold + '\\' + file[2]
#     #path = r'C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0\ngsim11.0.txt'
#     data = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
#     # name = os.path.splitext(file[0])[0]
#     cur_para = np.array(filefoldname,)
#
#
#     train_size = int(0.8 * len(data))
#     train = data.iloc[:train_size, :]
#     test = data.iloc[train_size:, :]