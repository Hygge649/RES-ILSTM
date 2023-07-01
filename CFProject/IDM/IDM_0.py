import math
import numpy as np
import pandas as pd
from sko.PSO import PSO
import matplotlib.pyplot as plt
import os


def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None


def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None


# D(x)
lb_x_idm = [0.500, 0.5, 21.7, 0.1, 5]
ub_x_idm = [4.500, 4.5, 30.7, 2, 10]

# 便利数据中的每一段跟驰数据
filefoldnames = r'C:\Users\SimCCAD\Desktop\train'
count = len(os.listdir(filefoldnames))   #177
# IDMfilefold = r'C:\Users\SimCCAD\Desktop\IDM\train'
parameter = pd.DataFrame(columns=['Maximum_Acc', 'Comfortable_Dec', 'Desire_Spe',
                                  'Desire_Spa_Tim', 'Minimum_Spa', 'loss(RMSE_ACC)'])
arr = np.zeros((1, 3))
idm_error = pd.DataFrame(arr, columns=['RMSE_A',  'RMSE_V', 'RMSE_GAP'], index=['train'])


def draw_picture_idm(p):
    x1, x2, x3, x4, x5 = p
    Maximum_Acc = x1  # 最大加速度
    Comfortable_Dec = x2  # 舒适减速度
    Desire_Spe = x3  # 期望速度
    Desire_Spa_Tim = x4  # 期望车头时距
    Minimum_Spa = x5  # 最短车头间距
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

    for i in range(len(data_train) - 1):

        cur_head_space = space_headway[i]
        cur_follower_speed = follower_speed[i]
        cur_follower_acc = follower_acc[i]
        cur_speed_diff = speed_diff[i]

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


def final_picture(data, sim_acc, sim_spe, sim_loc, sim_headway, filefold):

    x = data["Frame_ID"].values
    y1, y2, y3, y4 = sim_acc, sim_spe, sim_loc, sim_headway  # sim ---
    z1 = data["Mean_Acceleration"].values  # data
    z2 = data["Mean_Speed"].values
    z3 = data["LocalY"].values
    z4 = data["gap"].values

    # 'b', 'g', 'r', 'c', 'm', 'y', 'k'

    plt.title("IDM_a")
    plt.plot(x, y1, c='r', label='IDM')
    plt.plot(x, z1, c='g', label='Data')
    plt.legend()
    plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    plt.savefig(filefold + '/' + '/IDM_a_0.jpg')
    plt.show()

    plt.title("IDM_v")
    plt.plot(x, y2, c='r', label='IDM')
    plt.plot(x, z2, c='g', label='Data')
    plt.legend()
    plt.grid(color='k', visible=1, linestyle='--', linewidth=0.5)
    plt.savefig(filefold + '/' + '/IDM_v_0.jpg')
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
    plt.savefig(filefold + '/' + '/IDM_x_s_0.jpg')
    plt.show()


for filefoldname in os.listdir(filefoldnames):
    filefold = os.path.join(filefoldnames, filefoldname)
    # filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
    file = os.listdir(filefold)
    # path = filefold + '\\' + file[4]
    path = os.path.join(filefold, file[6])
    data_train = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')

    def function_idm(p):

        x1,x2,x3,x4,x5= p
        Maximum_Acc = x1  # 最大加速度
        Comfortable_Dec = x2  # 舒适减速度
        Desire_Spe = x3  # 期望速度
        Desire_Spa_Tim = x4  # 期望车头时距
        Minimum_Spa = x5  # 最短车头间距
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
            cur_speed_diff = speed_diff[i]

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

        return get_rmse(follower_acc,cur_sim_acc)

    pso_idm = PSO(func=function_idm, dim=5, pop=20, max_iter=20, lb=lb_x_idm, ub=ub_x_idm, w=0.8, c1=0.5, c2=0.5)

    #得到最优的x和y
    best_x_idm, best_y_idm = pso_idm.run()


    a, v, x = draw_picture_idm(best_x_idm)
    s = data_train.LocalY_leader[1:,] - data_train.Vehicle_length[1:,] - x


    IDM_data = pd.DataFrame(data=None,
                            columns=['Frame_ID', 'ACC', 'Mean_Speed',
                                     'Speed_Diff', 'LOC', 'gap'])
    IDM_data['Frame_ID'] = data_train.Frame_ID[1:,]
    IDM_data['ACC'] = a
    IDM_data['Mean_Speed'] = v
    IDM_data['Speed_Diff'] = data_train.Mean_Speed_leader[1:,] - v
    IDM_data['LOC'] = x
    IDM_data['gap'] = s

    # train_path = 'idm_0.txt'
    # train_path = os.path.join(filefold, train_path)
    # IDM_data.to_csv(train_path, sep='\t', index=False)

    cur_para = []
    cur_para.extend(best_x_idm)
    cur_para.extend(best_y_idm)
    cur_para = np.array(cur_para)
    cur_para = pd.DataFrame(cur_para.reshape(1, 6),
                            columns=['Maximum_Acc', 'Comfortable_Dec', 'Desire_Spe',
                                     'Desire_Spa_Tim', 'Minimum_Spa', 'loss(RMSE_ACC)'])

    # cur_path = 'idm_parameter.txt'
    # cur_path = os.path.join(filefold, cur_path)
    # cur_para.to_csv(cur_path, sep='\t', index=False)

    # parameter +=cur_para
    parameter = pd.concat([parameter, cur_para], axis=0)


    # IDM_ERROR:train,test
    def IDM_ERROR(pre, truth):

        rmse_a = get_rmse(pre.ACC, truth.Mean_Acceleration)
        rmse_v = get_rmse(pre.Mean_Speed, truth.Mean_Speed)
        rmse_gap =get_rmse(pre.gap, truth.gap)
        error = pd.DataFrame(data=[[rmse_a, rmse_v, rmse_gap]],
                             columns=['RMSE_A', 'RMSE_V', 'RMSE_GAP'],
                             index=['train'])
        return error


    train_error = IDM_ERROR(IDM_data, data_train.iloc[1:, ])
    # error_path = 'idm_0_error.txt'
    # IDM_ERROR_path = os.path.join(filefold, error_path)
    # train_error.to_csv(IDM_ERROR_path, sep='\t')

    idm_error += train_error



    # final_picture(data_train.iloc[1:, ], a, v, x, s, filefold)

    # 残差数据  DATA -SIM1

    def diffoutput(IDM_data, truth):

        diff_data = pd.DataFrame()
        diff_data['Frame_ID'] = truth['Frame_ID']
        diff_data['Speed'] = truth['Mean_Speed'] - IDM_data['Mean_Speed']
        diff_data['Speed_Diff'] = truth['speed_diff'] - truth['Mean_Speed_leader'] + IDM_data['Mean_Speed']
        diff_data['gap'] = truth['gap'] - IDM_data['gap']

        return diff_data


    # diff_train = diffoutput(IDM_data, data_train[['Frame_ID', 'Mean_Speed', 'Mean_Speed_leader', 'speed_diff', 'gap']])

    # diff_train_path = 'DIFF_train_0.txt'
    # diff_train_path = os.path.join(filefold, diff_train_path)
    # diff_train.to_csv(diff_train_path, sep='\t', index=False)
#
# parameter_path = r'C:\Users\SimCCAD\Desktop\idm_0_parameter.txt'
# parameter.to_csv(parameter_path, sep='\t', index=False)

idm_error = idm_error / count
print(idm_error)


