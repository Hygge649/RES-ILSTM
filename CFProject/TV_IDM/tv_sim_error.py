'''
SIM_TV_IDM
input: tv_parameters
output: 1.best_para for every trajectory:Maximum_Acc	Desire_Spe	Comfortable_Dec	Minimum_Spa	Desire_Spa_Tim	loss(RMSE_ACC)
2.loss:MSE_V	MSE_GAP	RMSE_V	RMSE_GAP
'''

import os
import pandas as pd
import numpy as np
import math

def IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta=4):
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        return item1 + max(0, item2 + item3)

    desired_S_n = desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n)
    a_n_t = a_max_n * (1 - (V_n_t / desired_V_n) ** beta - (desired_S_n / S_n_t) ** 2)

    return a_n_t

#J(MOP_SIM,MOP_DATA), J SET AS MSE AND RMSE
def mse_rmse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差, 均方根误差：是均方误差的算术平方根
    """
    if len(records_real) == len(records_predict):
        mse = sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
        rmse = math.sqrt(mse)

        return mse, rmse
    else:
        return None


def IDM_ERROR(pre, truth):
    mse_v, rmse_v = mse_rmse(pre.iloc[:, 0], truth.Mean_Speed)
    mse_gap, rmse_gap = mse_rmse(pre.iloc[:, 1], truth.gap)
    error = pd.DataFrame(data=[[mse_v, mse_gap, rmse_v, rmse_gap]],
                         columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP']
                        )
    return error

filefoldnames = r'F:\train'
count = len(os.listdir(filefoldnames))
tv_parameter = pd.DataFrame(columns=['Maximum_Acc',  'Desire_Spe', 'Comfortable_Dec',
                                  'Minimum_Spa', 'Desire_Spa_Tim', 'loss(RMSE_ACC)'])

arr = np.zeros((1, 4))
tv_idm_error = pd.DataFrame(arr, columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'])


for filefoldname in os.listdir(filefoldnames):
    filefold = os.path.join(filefoldnames, filefoldname)
    # filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
    file = os.listdir(filefold)
    # path = filefold + '\\' + file[4]
    path = os.path.join(filefold, file[0])
    data = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
    path_2 = os.path.join(filefold, file[1])
    tv_params = pd.read_csv(path_2, header=None, delim_whitespace=True, encoding='utf-8')
    #use para for every frame to generate a sim trajectory

    args = pd.DataFrame(data=[data.speed_diff, data.gap, data.Mean_Speed]).T
    args = pd.concat([args.iloc[:-1, :], tv_params], axis=1)
    args = pd.concat([args, data[['LocalY', 'LocalY_leader', 'Vehicle_length']]], axis=1)

    a = []
    v = []
    gap = []
    for frame in args.itertuples():
        frame = np.array(frame)
        a_n_t_hat = IDM_cf_model_for_p(frame[1], frame[2], frame[3],
                                       frame[4], frame[5], frame[6], frame[7], frame[8]
                                       )
        t = 0.1
        cur_v = frame[3] + a_n_t_hat * t
        cur_x = frame[9] + cur_v * t - (a_n_t_hat * t * t) / 2
        cur_gap = frame[10] - cur_x - frame[11]

        v.append(cur_v)
        gap.append(cur_gap)

    # return sim_v,gap
    sim_tv = pd.DataFrame(data=[v, gap]).T
    cur_tv_idm_error = IDM_ERROR(sim_tv.iloc[:-1, :], data[['Mean_Speed', 'gap']].iloc[1:, :])
    tv_idm_error = pd.concat([tv_idm_error, cur_tv_idm_error], axis=0, ignore_index=True)

tv_error = pd.DataFrame(data=tv_idm_error.iloc[1:, :],
                             columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'])
np.savetxt(r'G:\result\tv_error.csv', tv_error)


