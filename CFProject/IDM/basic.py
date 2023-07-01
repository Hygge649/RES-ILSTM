import pandas as pd
import numpy as np
import math
import os
import matplotlib.pyplot as plt

#load data: data_truth, data_diff, data_idm
filefold= r'C:\Users\SimCCAD\Desktop\IDM\train'

count = 0
arr1 = np.zeros((1, 2))
arr2 = np.zeros((3,2))
idm_loss = pd.DataFrame(arr1, columns=['RMSE_V', 'RMSE_GAP'], index=['train'])#, 'test'])
# idm_loss_1 = pd.DataFrame(arr1, columns=['RMSE_V', 'RMSE_GAP'], index=['train', 'test'])
# lstm_loss = pd.DataFrame(arr2, columns=['RMSE_V', 'RMSE_GAP'], index=['train', 'valid', 'test'])
# # lstm_loss_1 = pd.DataFrame(arr2, columns=['RMSE_V', 'RMSE_GAP'], index=['train', 'valid', 'test'])
# diff_idm_loss = pd.DataFrame(arr2, columns=['RMSE_V', 'RMSE_GAP'], index=['train', 'valid', 'test'])
# diff_idm_loss_1 = pd.DataFrame(arr2, columns=['RMSE_V', 'RMSE_GAP'], index=['train', 'valid', 'test'])
# diff_idm_loss_2 = pd.DataFrame(arr2, columns=['RMSE_V', 'RMSE_GAP'], index=['train', 'valid', 'test'])

for file in os.listdir(filefold):
    path = os.path.join(filefold, file)
    train = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
    # filefold = 'C:\\Users\\SimCCAD\\Desktop\\IDM' + '\\' + filefoldname
    # filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
    file = os.listdir(filefold)
    count +=1

    idm_error = pd.read_csv(filefold + '\\' + file[2], delim_whitespace=True, encoding='utf-8')
    # idm_error_1 = pd.read_csv(filefold + '\\' + file[11], delim_whitespace=True, encoding='utf-8')
    # # lstm_error = pd.read_csv(filefold + '\\' + file[11], delim_whitespace=True, encoding='utf-8')
    # # # # lstm_error_1 = pd.read_csv(filefold + '\\' + file[8], delim_whitespace=True, encoding='utf-8')
    # diff_idm_error = pd.read_csv(filefold + '\\' + file[5], delim_whitespace=True, encoding='utf-8')
    # diff_idm_error_1 = pd.read_csv(filefold + '\\' + file[6], delim_whitespace=True, encoding='utf-8')
    # diff_idm_error_2 = pd.read_csv(filefold + '\\' + file[7], delim_whitespace=True, encoding='utf-8')
    #

    idm_loss += idm_error
    # idm_loss_1 += idm_error_1
    # # lstm_loss += lstm_error
    # # # # lstm_loss_1 += lstm_error_1
    # diff_idm_loss += diff_idm_error
    # diff_idm_loss_1 += diff_idm_error_1
    # diff_idm_loss_2 += diff_idm_error_2
    #


idm_loss = idm_loss / count
# idm_loss_1 = idm_loss_1 / count
# # lstm_loss = lstm_loss / count
# # # # lstm_loss_1 = lstm_loss_1 / count
# diff_idm_loss = diff_idm_loss /count
# diff_idm_loss_1 = diff_idm_loss_1 /count
# diff_idm_loss_2 = diff_idm_loss_2 /count
