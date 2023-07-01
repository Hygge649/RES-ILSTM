import os
import numpy as np
import pandas as pd
import math

def draw_picture_idm(p):
    # input:x,data_step
    # output: MOP of each window
    # x1, x2, x3, x4, x5 = p

    Maximum_Acc = p[0, 0]  # 最大加速度
    Desire_Spe = p[0, 1]  # 期望速度
    Comfortable_Dec = p[0, 2]  # 舒适减速度
    Minimum_Spa = p[0, 3]  # 最短车头间距
    Desire_Spa_Tim = p[0, 4]  # 期望车头时距
    Para_Beta = 4 # 加速度系数. fixed as 4

    simulated_spe = []
    simulated_loc = []
    cur_sim_acc = []


    follower_position = data["LocalY"].values
    follower_speed = data["Mean_Speed"].values
    speed_diff = data["speed_diff"].values
    space_headway = data["gap"].values  # gap, v.diff
    follower_acc = data["Mean_Acceleration"].values

    cur_sim_spe = follower_speed[0]
    cur_sim_loc = follower_position[0]
    t = 0.1  # 1 frame = 0.1s

    for i in range(len(data)):

        cur_head_space = space_headway[i]
        cur_follower_speed = follower_speed[i]
        cur_follower_acc = follower_acc[i]
        if i == 0:
            cur_follower_acc_last = follower_acc[i]
        else:
            cur_follower_acc_last = follower_acc[i - 1]
        cur_speed_diff = speed_diff[i]

        # if ((cur_follower_speed<1) or (abs(cur_follower_acc)<1)):
        if ((cur_follower_speed < 1) or ((cur_follower_acc) - (cur_follower_acc_last) < 0.5)):
            cur_follow_1_acc = cur_follower_acc
        else:
            try:
                cur_des_spa = Minimum_Spa + max(0, cur_follower_speed * Desire_Spa_Tim + (
                        cur_follower_speed * cur_speed_diff \
                        / (2 * math.sqrt(Maximum_Acc * Comfortable_Dec))))  # S*,
            except:
                cur_des_spa = sum(space_headway) / len(space_headway)  # space_headway.mean

            cur_follow_1_acc = Maximum_Acc * (1 - math.pow((cur_follower_speed / Desire_Spe), Para_Beta) -
                                              math.pow(cur_des_spa / cur_head_space,
                                                       2))  # an

        cur_sim_spe = cur_sim_spe + cur_follow_1_acc * t  # v
        cur_sim_loc = cur_sim_loc + cur_sim_spe * t - (cur_follow_1_acc * t * t) / 2  # x

        cur_sim_acc.append(cur_follow_1_acc)
        simulated_spe.append(cur_sim_spe)
        simulated_loc.append(cur_sim_loc)


    return cur_sim_acc, simulated_spe, simulated_loc

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
min_parameter = np.zeros((1, 5))

arr = np.zeros((1, 4))
min_idm_error = pd.DataFrame(arr, columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'])



for filefoldname in os.listdir(filefoldnames):
    filefold = os.path.join(filefoldnames, filefoldname)
    file = os.listdir(filefold)
    path = os.path.join(filefold, file[0])
    data = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
    path_1 = os.path.join(filefold, file[-1])
    using_all = np.loadtxt(path_1).reshape(1,-1)

    min_parameter = np.concatenate([min_parameter, using_all], axis=0)

    _, min_v,  min_x = draw_picture_idm(using_all)
    gap = data.LocalY_leader - min_x - data.Vehicle_length
    sim_min = pd.DataFrame(data=[min_v, gap]).T
    cur_min_idm_error = IDM_ERROR(sim_min, data[['Mean_Speed',  'gap']])
    min_idm_error = pd.concat([min_idm_error, cur_min_idm_error], axis=0, ignore_index=True)

min_parameter = pd.DataFrame(data=min_parameter[1:, :],
                             columns=['Maximum_Acc',  'Desire_Spe', 'Comfortable_Dec',
                                      'Minimum_Spa',  'Desire_Spa_Tim'])
# np.savetxt(r'F:\result\parameter_min.csv', min_parameter)

min_idm_error = pd.DataFrame(data=min_idm_error.iloc[1:, :],
                             columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'])
# np.savetxt(r'F:\result\idm_error_min.csv', min_idm_error)

