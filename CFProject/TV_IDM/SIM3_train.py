import os
import pandas as pd
import numpy as np

def IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta=4):
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        return item1 + max(0, item2 + item3)

    desired_S_n = desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n)
    a_n_t = a_max_n * (1 - (V_n_t / desired_V_n) ** beta - (desired_S_n / S_n_t) ** 2)

    return a_n_t

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


def diffoutput(IDM_data, truth):
    diff_data = pd.DataFrame()
    # diff_data['Frame_ID'] = truth['Frame_ID']
    diff_data['Residual_Speed'] = truth['Mean_Speed'] - IDM_data['Mean_Speed']
    # diff_data['Speed_Diff'] = truth['speed_diff'] - truth['Mean_Speed_leader'] + IDM_data['Mean_Speed']
    diff_data['Residual_gap'] = truth['gap'] - IDM_data['gap']

    return diff_data

filefoldnames = r'G:\1\train'
count = len(os.listdir(filefoldnames))   #174
# IDMfilefold = r'C:\Users\SimCCAD\Desktop\IDM\train'
parameter = pd.DataFrame(columns=['Maximum_Acc',  'Desire_Spe', 'Comfortable_Dec',
                                  'Minimum_Spa', 'Desire_Spa_Tim', 'loss(RMSE_ACC)'])
arr = np.zeros((1, 4))
idm_error = pd.DataFrame(columns=['MSE_V', 'MSE_GAP', 'RMSE_V', 'RMSE_GAP'])


for filefoldname in os.listdir(filefoldnames):
    filefold = os.path.join(filefoldnames, filefoldname)
    file = os.listdir(filefold)
    path = os.path.join(filefold, file[1])
    train = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')

    name = os.path.splitext(file[1])[0]

    path_para = os.path.join(filefold, file[4])
    paras = pd.read_csv(path_para, header=None, delim_whitespace=True, encoding='utf-8')
    paras.columns = ["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]
    sim_acc, sim_spe, sim_loc, sim_headway = [], [], [], []
    state = train[['speed_diff', 'gap', 'Mean_Speed', 'LocalY', 'LocalY_leader', 'Vehicle_length']]
    state = state.iloc[:-1, :]
    args = pd.concat([state, paras], axis=1)
    t = 0.1
    for arg in args.values:
        [delta_V_n_t, S_n_t, V_n_t, LocalY, LocalY_leader, Vehicle_length,\
            a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n] = arg
        a = IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n)
        v = V_n_t + a * t
        x = LocalY + v * t - (a * t * t) / 2
        s = LocalY_leader -Vehicle_length - x

        sim_acc.append(a)
        sim_spe.append(v)
        sim_loc.append(x)
        sim_headway.append(s)

    sim3 = output(sim_acc, sim_spe, sim_loc, sim_headway, train.iloc[1:, :])
    # train_path = 'idm_sim3.txt'
    # train_path = os.path.join(filefold, train_path)
    # sim3.to_csv(train_path, sep='\t', index=False)
    #
    diff_sim3 = diffoutput(sim3, train.iloc[1:,:])
    #
    # diff_train_path = 'DIFF_train_sim3.txt'
    # diff_train_path = os.path.join(filefold, diff_train_path)
    # diff_sim3.to_csv(diff_train_path, sep='\t', index=False)

    sim3 = pd.concat([sim3, diff_sim3], axis=1)
    sim3 = pd.concat([sim3, train.Mean_Acceleration], axis=1)
    sim3_path = 'sim3.txt'
    sim3_path = os.path.join(filefold, sim3_path)
    sim3.to_csv(sim3_path, sep='\t', index=False)

    print(name)


