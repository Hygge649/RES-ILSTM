import math
import numpy as np
import pandas as pd
from sko.PSO import PSO
import matplotlib.pyplot as plt
import os

def create_dataset(dataset, window_size):
    data_step = []
    for i in range(math.ceil((len(dataset) / window_size))):
        a = dataset.iloc[i*window_size:(i*window_size + window_size), :]
        data_step.append(a)
    return data_step

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

# D(x)
# lb_x_idm = [0.500, 0.5, 21.7, 0.1, 5]
# ub_x_idm = [4.500, 4.5, 30.7, 2, 10]

lb_x_idm = [0.1, 1, 0.1, 0.1, 0.1]
ub_x_idm = [2.5, 40, 5, 10, 5]

def function_idm(p):
    x1, x2, x3, x4, x5 = p

    Maximum_Acc = x1  # 最大加速度
    Desire_Spe = x2  # 期望速度
    Comfortable_Dec = x3  # 舒适减速度
    Minimum_Spa = x4  # 最短车头间距
    Desire_Spa_Tim = x5  # 期望车头时距
    Para_Beta = 4  # 加速度系数. fixed as 4


    simulated_spe = []
    simulated_loc = []
    cur_sim_acc = []

    follower_position = data_train["LocalY"].values
    follower_speed = data_train["Mean_Speed"].values
    speed_diff = data_train["speed_diff"].values
    space_headway = data_train["gap"].values  # gap, v.diff
    follower_acc = data_train["Mean_Acceleration"].values

    cur_sim_spe = follower_speed[0]
    cur_sim_loc = follower_position[0]
    t = 0.1  # 1 frame = 0.1s

    for i in range(len(data_train)):

        cur_head_space = space_headway[i]
        cur_follower_speed = follower_speed[i]
        cur_follower_acc = follower_acc[i]
        if i == 0:
            cur_follower_acc_last = follower_acc[i]
        else:
            cur_follower_acc_last = follower_acc[i - 1]
        cur_speed_diff = speed_diff[i]

        # # if v<0,keep on
        # # then IDM
        # # if cur_follower_speed<1.1: #19.46493688320361
        # # if ((cur_follower_speed<1) or (abs(cur_follower_acc)<1)): # 19.237350933207246
        if ((cur_follower_speed < 1) or ((cur_follower_acc) - (cur_follower_acc_last) < 0.5)):
            # if cur_follower_speed<0.9: #19.631535239728976
            # if cur_follower_speed <0.8: #20.10880045743511
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


    # MIN J
    mse_a, rmse_a = mse_rmse(follower_acc, cur_sim_acc)
    mse_v, rmse_v = mse_rmse(follower_speed, simulated_spe)

    return rmse_a

def draw_picture_idm(p, data_step):
    # input:x,data_step
    # output: MOP of each window
    x1, x2, x3, x4, x5 = p

    Maximum_Acc = x1  # 最大加速度
    Desire_Spe = x2  # 期望速度
    Comfortable_Dec = x3  # 舒适减速度
    Minimum_Spa = x4  # 最短车头间距
    Desire_Spa_Tim = x5  # 期望车头时距
    Para_Beta = 4  # 加速度系数. fixed as 4

    simulated_spe = []
    simulated_loc = []
    cur_sim_acc = []

    data = data_step

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

def output(sim_acc, sim_spe, sim_loc, sim_headway, truth):
    IDM_data = pd.DataFrame(data=None,
                            columns=['Frame_ID', 'ACC', 'Mean_Speed',
                                     'Speed_Diff', 'LOC', 'gap'])
    IDM_data['Frame_ID'] = truth['Frame_ID']
    IDM_data['ACC'] = sim_acc
    IDM_data['Mean_Speed'] = sim_spe
    IDM_data['Speed_Diff'] = truth['Mean_Speed_leader'] - sim_spe
    IDM_data['LOC'] = sim_loc
    IDM_data['gap'] = sim_headway

    return IDM_data


filefoldnames = r'G:\1\train'
count = len(os.listdir(filefoldnames))   #174
# IDMfilefold = r'C:\Users\SimCCAD\Desktop\IDM\train'
parameter = pd.DataFrame(columns=['Maximum_Acc',  'Desire_Spe', 'Comfortable_Dec',
                                  'Minimum_Spa', 'Desire_Spa_Tim', 'loss(RMSE_ACC)'])
arr = np.zeros((1, 4))
idm_error = pd.DataFrame(columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'])


for filefoldname in os.listdir(filefoldnames):
    filefold = os.path.join(filefoldnames, filefoldname)
    # filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
    file = os.listdir(filefold)
    # path = filefold + '\\' + file[4]
    path = os.path.join(filefold, file[1])
    train = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
    # train = train[["Frame_ID", "Mean_Speed", 'Mean_Speed_leader', 'LocalY',
    #                'LocalY_leader',  'Vehicle_length', "speed_diff", "gap"]]
    name = os.path.splitext(file[1])[0]


    window_size = 50
    data_step = create_dataset(train, window_size)
    best_x_pso, best_y_pso = [], []
    sim_acc, sim_spe, sim_loc, sim_headway = [], [], [], []

    for data_train in data_step:

        pso_idm = PSO(func=function_idm, dim=5, pop=20, max_iter=20, lb=lb_x_idm, ub=ub_x_idm, w=0.8, c1=0.5, c2=0.5)
        best_x_pso_step, best_y_pso_step = pso_idm.run()

        best_x_pso.append(best_x_pso_step)
        best_y_pso.append(best_y_pso_step)


        a, v, x = draw_picture_idm(best_x_pso_step, data_train)
        s = data_train["LocalY_leader"] - data_train["Vehicle_length"] - x
        # train
        sim_acc.extend(a)
        sim_spe.extend(v)
        sim_loc.extend(x)
        sim_headway.extend(s)


    IDM_TRAIN = output(sim_acc, sim_spe, sim_loc, sim_headway, train)
    # train_path = 'idm.txt'
    # train_path = os.path.join(filefold, train_path)
    # IDM_TRAIN.to_csv(train_path, sep='\t', index=False)

    # best_x_pso = np.mean(best_x_pso, axis=0)
    # best_y_pso = np.mean(best_y_pso).reshape(1,)
    # best_y_pso = np.array(best_y_pso)


    # cur_para = []
    # cur_para.extend(best_x_pso)
    # cur_para.extend(best_y_pso)
    # cur_para = np.array(cur_para)
    # cur_para = pd.DataFrame(cur_para.reshape(1, 6),
    #                         columns=['Maximum_Acc',  'Desire_Spe', 'Comfortable_Dec',
    #                                  'Minimum_Spa',  'Desire_Spa_Tim', 'loss(RMSE_ACC)'])
    # cur_path = 'using_all_data_window_size.txt'
    # cur_path = os.path.join(filefold, cur_path)
    # cur_para.to_csv(cur_path, sep='\t', index=False)
    # # parameter +=cur_para
    # parameter = pd.concat([parameter, cur_para], axis=0, ignore_index=True)

    # IDM_ERROR:train,test
#     def IDM_ERROR(pre, truth):
#         mse_v, rmse_v = mse_rmse(pre.Mean_Speed, truth.Mean_Speed)
#         mse_gap, rmse_gap = mse_rmse(pre.gap, truth.gap)
#         error = pd.DataFrame(data=[[mse_v, mse_gap, rmse_v, rmse_gap]],
#                              columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'],
#                             )
#         return error
#
#     train_error = IDM_ERROR(IDM_TRAIN, train[['Mean_Speed',  'gap']])
#     idm_error = pd.concat([idm_error, train_error], axis=0, ignore_index=True)
#
#
# print(idm_error['RMSE_GAP'].mean())
# print(idm_error['RMSE_GAP'].var())


# np.savetxt(r'F:\result\window_size_parameter.csv', parameter)
#
#
# np.savetxt(r'F:\result\window_size_idm_error.csv', idm_error)
    # idm_error += train_error

    # 残差数据  DATA -SIM1

    def diffoutput(IDM_data, truth):

        diff_data = pd.DataFrame()
        # diff_data['Frame_ID'] = truth['Frame_ID']
        diff_data['Residual_Speed'] = truth['Mean_Speed'] - IDM_data['Mean_Speed']
        # diff_data['Speed_Diff'] = truth['speed_diff'] - truth['Mean_Speed_leader'] + IDM_data['Mean_Speed']
        diff_data['Residual_gap'] = truth['gap'] - IDM_data['gap']

        return diff_data
    diff_train = diffoutput(IDM_TRAIN, train[['Frame_ID', 'Mean_Speed', 'Mean_Speed_leader', 'speed_diff', 'gap']])
    # diff_train_path = 'DIFF_train.txt'
    # diff_train_path = os.path.join(filefold, diff_train_path)
    # diff_train.to_csv(diff_train_path, sep='\t', index=False)

    acc = train.Mean_Acceleration
    sim2 = pd.concat([IDM_TRAIN, diff_train], axis=1)
    sim2 = pd.concat([sim2, acc], axis=1)
    sim2_path = 'sim2.txt'
    sim2_path = os.path.join(filefold, sim2_path)
    sim2.to_csv(sim2_path, sep='\t', index=False)

    print(name)
# parameter_path = r'C:\Users\SimCCAD\Desktop\idm_parameter.txt'
# parameter.to_csv(parameter_path, sep='\t', index=False)

# idm_error = idm_error/count
# print(idm_error)
#

'''
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.042585  0.660598  0.137987  0.758490
test   0.131710  2.810674  0.208769  1.280959


          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.098595  1.921067  0.152694  0.849786

'''




from pathlib import Path
pic_path = Path('./idm_picture_1min')

def final_picture(data, sim_acc, sim_spe, sim_loc, sim_headway):

    x = data["Frame_ID"].values
    y1, y2, y3, y4 = sim_acc, sim_spe, sim_loc, sim_headway  # sim ---
    z1 = data["Mean_Acceleration"].values  # data
    z2 = data["Mean_Speed"].values
    z3 = data["LocalY"].values
    z4 = data["gap"].values

    # 'b', 'g', 'r', 'c', 'm', 'y', 'k'

    # plt.title("IDM_a")
    # plt.plot(x, y1, c='r', label='IDM')
    # plt.plot(x, z1, c='g', label='Data')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/' + 'IDM_ACC.jpg')
    # plt.show()

    plt.title("IDM_v")
    plt.plot(x, y2, c='r', label='IDM')
    plt.plot(x, z2, c='g', label='Data')
    plt.legend()
    plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    plt.savefig(pic_path + '/IDM_V.jpg')
    plt.show()

    fig, ax1 = plt.subplots()
    plt.title("IDM_x,s")

    plt.plot(x, y3, c='b', linestyle='--')
    plt.plot(x, z3, c='b', label='data_s')  # x
    ax1.set_ylabel('loc', color='b')
    plt.legend()
    ax2 = ax1.twinx()  # 创建共用x轴的第二
    plt.plot(x, y4, c='m', linestyle='--')
    plt.plot(x, z4, c='m', label='data_x')  # s
    ax2.set_ylabel('Gap', color='m')
    plt.legend()
    plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    plt.savefig(pic_path + '/IDM_x_s.jpg')
    plt.show()

    # plt.title("IDM_x_v")
    # plt.plot(z3, z2, c='r', label='Data')  # data
    # plt.plot(y3, y2, c='g', label='IDM')  # idm
    # plt.xlabel('x')
    # plt.ylabel('v')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/' + 'IDM_x_v.jpg')
    # plt.show()
    #
    # plt.title("IDM_gap_v")
    # plt.plot(z4, z2, c='r', label='Data')  # data
    # plt.plot(y4, y2, c='g', label='IDM')  # idm
    # plt.xlabel('gap')
    # plt.ylabel('v')
    # plt.legend()
    # plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    # plt.savefig(filefold + '/' + 'IDM_gap_v.jpg')
    # plt.show()

import seaborn as sns
def ploterror(error):

    sns.kdeplot(error['RMSE_V'], fill=True)
    plt.title("RMSE_V")
    plt.show()

    # sns.boxplot(parameter.iloc[:, 1],  notch=True)
    sns.kdeplot(error['RMSE_GAP'], fill=True)
    plt.title("RMSE_GAP")
    plt.show()