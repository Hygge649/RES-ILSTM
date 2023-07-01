import math
import numpy as np
import pandas as pd
# from sko.PSO import PSO
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


def draw_picture_idm(p, data_step):
    # input:x,data_step
    # output: MOP of each window
    x1, x2, x3, x4, x5 = p

    Maximum_Acc = x1  # 最大加速度
    Comfortable_Dec = x2  # 舒适减速度
    Desire_Spe = x3  # 期望速度
    Desire_Spa_Tim = x4  # 期望车头时距
    Minimum_Spa = x5  # 最短车头间距
    Para_Beta = 4  # 加速度系数

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

# IDM_ERROR:train,test
def IDM_ERROR(pre, truth):
    mse_v, rmse_v = mse_rmse(pre.Mean_Speed, truth.Mean_Speed)
    mse_gap, rmse_gap = mse_rmse(pre.gap, truth.gap)
    error = pd.DataFrame(data=[[mse_v, mse_gap, rmse_v, rmse_gap]],
                         columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'],
                         index=['test'])
    return error

def diffoutput(IDM_data, truth):

    diff_data = pd.DataFrame()
    diff_data['Frame_ID'] = truth['Frame_ID']
    diff_data['Residual_Speed'] = truth['Mean_Speed'] - IDM_data['Mean_Speed']
    # diff_data['Speed_Diff'] = truth['speed_diff'] - truth['Mean_Speed_leader'] + IDM_data['Mean_Speed']
    diff_data['Residual_gap'] = truth['gap'] - IDM_data['gap']

    return diff_data

filefoldnames = r'G:\test'
count = len(os.listdir(filefoldnames))   #
# IDMfilefold = r'C:\Users\SimCCAD\Desktop\IDM\train'
parameter = pd.DataFrame(columns=['Maximum_Acc', 'Comfortable_Dec', 'Desire_Spe',
                                  'Desire_Spa_Tim', 'Minimum_Spa', 'loss(RMSE_ACC)'])
arr = np.zeros((1, 4))
idm_error = pd.DataFrame(arr, columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'], index=['test'])

#sim2
# parameter_path = r'G:\result\parameter_window_size.csv'
# parameter = pd.read_csv(parameter_path)#,  delim_whitespace=True, encoding='utf-8')
# best_x_pso= []
# best_x_pso.append(parameter['Maximum_Acc'].mean())
# best_x_pso.append(parameter['Comfortable_Dec'].mean())
# best_x_pso.append(parameter['Desire_Spe'].mean())
# best_x_pso.append(parameter['Desire_Spa_Tim'].mean())
# best_x_pso.append(parameter['Minimum_Spa'].mean())

#sim3
best_x_pso = pd.read_csv('../TV_IDM/para.csv', index_col=None)
best_x_pso = best_x_pso['0'].values.tolist()

for filefoldname in os.listdir(filefoldnames):
    filefold = filefoldnames + '\\' + filefoldname
    # filefold = C:\Users\SimCCAD\Desktop\NGSIM\ngsim11.0
    file = os.listdir(filefold)
    path = os.path.join(filefold, file[0])
    test = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
    name = os.path.splitext(file[0])[0]
    #
    # window_size = 50
    # data_step = create_dataset(test, window_size)
    sim_acc, sim_spe, sim_loc, sim_headway = [], [], [], []
    IDM_TEST = pd.DataFrame(data=None,
                            columns=['Frame_ID', 'ACC', 'Mean_Speed',
                                     'Speed_Diff', 'LOC', 'gap'])
    #
    # for data_test in data_step:
    #     # test
    sim_test_acc, sim_test_spe, sim_test_loc, sim_test_headway = [], [], [], []
    a_test, v_test, x_test = draw_picture_idm(best_x_pso, test)
    s_test = test["LocalY_leader"] - test["Vehicle_length"] - x_test
    IDM_TEST = output(a_test, v_test, x_test, s_test, test)
    # sim_test_acc.extend(a_test)
    # sim_test_spe.extend(v_test)
    # sim_test_loc.extend(x_test)
    # sim_test_headway.extend(s_test)

    # IDM_TEST = output(sim_test_acc, sim_test_spe, sim_test_loc, sim_test_headway, test)
    # IDM_TEST = pd.concat([IDM_TEST, cur_IDM_TEST], axis=0)

    # test_path = 'idm.txt'
    # test_path = os.path.join(filefold, test_path)
    # IDM_TEST.to_csv(test_path, sep='\t', index=False)


    # test_error = IDM_ERROR(IDM_TEST, test[['Mean_Speed',  'gap']])
    # error_path = 'idm_error.txt'
    # IDM_ERROR_path = os.path.join(filefold, error_path)
    # test_error.to_csv(IDM_ERROR_path, sep='\t')
    #
    # idm_error += test_error

    #残差数据  DATA -SIM1
    diff_train = diffoutput(IDM_TEST, test[['Frame_ID', 'Mean_Speed', 'Mean_Speed_leader', 'speed_diff', 'gap']])
    # diff_train_path = 'DIFF_test.txt'
    # diff_train_path = os.path.join(filefold, diff_train_path)
    # diff_train.to_csv(diff_train_path, sep='\t', index=False)

    # sim2_test = pd.concat([IDM_TEST, diff_train], axis=1)
    # # sim2_test_path = 'sim2_test.txt'
    # sim2_test_path = os.path.join(filefold, sim2_test_path)
    # sim2_test.to_csv(sim2_test_path, sep='\t', index=False)

    sim3_test = pd.concat([IDM_TEST, diff_train], axis=1)
    sim3_test_path = 'sim3_test.txt'
    sim3_test_path = os.path.join(filefold, sim3_test_path)
    sim3_test.to_csv(sim3_test_path, sep='\t', index=False)
    print(name)

# idm_error = idm_error/count
# print(idm_error)


'''
for one pair:
          MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.042585  0.660598  0.137987  0.758490
test   0.131710  2.810674  0.208769  1.280959


for all trajectory:
         MSE_V   MSE_GAP    RMSE_V  RMSE_GAP
train  0.09778  1.920565  0.152173  0.849595

         MSE_V    MSE_GAP    RMSE_V  RMSE_GAP
test  1.542872  57.225518  0.718114  3.034383
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