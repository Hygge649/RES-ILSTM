from typing import Dict, List, Union, Any
import numpy as np
import pickle, time, copy, os
import pylab
import threading, sys
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import random
from pathos.multiprocessing import ProcessingPool as Pool
import pandas as pd
from sko.PSO import PSO
import math
import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")



lb_x_idm = [0.1, 1, 0.1, 0.1, 0.1]
ub_x_idm = [2.5, 40, 5, 10, 5]

def set_cons(a_max_n_boundary=[0.1, 2.5], desired_V_n_boundary=[1, 40], a_comf_n_boundary=[0.1, 5],
             S_jam_boundary=[0.1, 10], desired_T_n_boundary=[0.1, 5],):# beta_boundary=[4, 4]):
# def set_cons(a_max_n_boundary=[0.5, 4.5], desired_V_n_boundary=[21.7, 30.7], a_comf_n_boundary=[0.5, 4.5],
#               S_jam_boundary=[0.1, 2], desired_T_n_boundary=[5, 10], beta_boundary=[4, 4]):
    a_max_n_boundary = a_max_n_boundary
    desired_V_n_boundary = desired_V_n_boundary
    a_comf_n_boundary = a_comf_n_boundary
    S_jam_boundary = S_jam_boundary
    desired_T_n_boundary = desired_T_n_boundary
    # beta_boundary = beta_boundary

    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - a_max_n_boundary[0]},
            {'type': 'ineq', 'fun': lambda x: -x[0] + a_max_n_boundary[1]},
            {'type': 'ineq', 'fun': lambda x: x[1] - desired_V_n_boundary[0]},
            {'type': 'ineq', 'fun': lambda x: -x[1] + desired_V_n_boundary[1]},
            {'type': 'ineq', 'fun': lambda x: x[2] - a_comf_n_boundary[0]},
            {'type': 'ineq', 'fun': lambda x: -x[2] + a_comf_n_boundary[1]},
            {'type': 'ineq', 'fun': lambda x: x[3] - S_jam_boundary[0]},
            {'type': 'ineq', 'fun': lambda x: -x[3] + S_jam_boundary[1]},
            {'type': 'ineq', 'fun': lambda x: x[4] - desired_T_n_boundary[0]},
            {'type': 'ineq', 'fun': lambda x: -x[4] + desired_T_n_boundary[1]})#,
            # {'type': 'ineq', 'fun': lambda x: x[5] - beta_boundary[0]},
            # {'type': 'ineq', 'fun': lambda x: -x[5] + beta_boundary[1]})
    return cons

def initialize(a_max_n_boundary=[0.1, 2.5], desired_V_n_boundary=[1, 40], a_comf_n_boundary=[0.1, 5],
               S_jam_boundary=[0.1, 10], desired_T_n_boundary=[0.1, 5], beta_boundary=[4, 4]):

# def initialize(a_max_n_boundary=[0.5, 4.5], desired_V_n_boundary=[21.7, 30.7], a_comf_n_boundary=[0.5, 4.5],
#              S_jam_boundary=[0.1, 2], desired_T_n_boundary=[5, 10], beta_boundary=[4, 4]):
    # a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    x0 = (random.uniform(a_max_n_boundary[0], a_max_n_boundary[1]),
          random.uniform(desired_V_n_boundary[0], desired_V_n_boundary[1]),
          random.uniform(a_comf_n_boundary[0], a_comf_n_boundary[1]),
          random.uniform(S_jam_boundary[0], S_jam_boundary[1]),
          random.uniform(desired_T_n_boundary[0], desired_T_n_boundary[1]), 4)
    return x0


def IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta=4):
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):  # s*
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        return item1 + max(0, item2 + item3)

    a_n_t = []
    for i in range(len(delta_V_n_t)):
        desired_S_n = desired_space_hw(S_jam_n, V_n_t[i], desired_T_n, delta_V_n_t[i], a_max_n, a_comf_n)
        a_n_t.append(a_max_n * (1 - (V_n_t[i] / desired_V_n) ** beta - (desired_S_n / S_n_t[i]) ** 2))

    return np.array(a_n_t)

def sim_new_spacing_l(new_pre_y, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
    res_spacing = []
    for i in range(len(new_pre_y)):
        res_spacing.append(new_pre_y[i] - old_ego_y[i] -
                           (2 * V_n_t[i] + a_n_t[i] * delta_t) / 2 * delta_t)
    return res_spacing

def all_rmse_using_fixed_params(args, x):
    S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_y, ego_y = args
    err = mean_squared_error(S_n_t_1, sim_new_spacing_l(next_pre_y, ego_y, V_n_t,
                                                        IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], )))
    return math.sqrt(err)

def obj_func(args):  # mse_gap
    S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_y, ego_y = args
    err = lambda x: mean_squared_error(S_n_t_1, sim_new_spacing_l(next_pre_y,  ego_y, V_n_t,
                                                                  IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1],
                                                                               x[2], x[3], x[4])))
    return err

def create_dataset(dataset, window_size):
    data_step = []
    for i in range(math.ceil((len(dataset) / window_size))):
        a = dataset.iloc[i*window_size:(i*window_size + window_size), :]
        data_step.append(a)
    return data_step

def min_gap(id, cf_data):

    # cf_data = create_dataset(cf_data, 50)

    data_step = []
    window_size = 50
    for i in range(math.ceil((len(cf_data) / window_size))):
        a = cf_data.iloc[i * window_size:(i * window_size + window_size), :]
        data_step.append(a)

    cf_data = data_step


    best_x_pso, best_y_pso = [], []

    for data_step in cf_data:
        if len(data_step) ==1:
            best_x_pso_step = best_x_pso_step
        else:

            delta_V_n_t = data_step.speed_diff[:-1].values
            S_n_t = data_step.gap[:-1].values
            V_n_t = data_step.Mean_Speed[:-1].values
            a = data_step.Mean_Acceleration[:-1].values
            S_n_t_y = data_step.gap[1:].values
            ego_y = data_step.LocalY[:-1].values
            next_pre_y = data_step.LocalY_leader[1:].values
            frame_id = data_step.Frame_ID[:-1].values

            args = (np.array(S_n_t_y), np.array(delta_V_n_t), np.array(S_n_t),
                    np.array(V_n_t), np.array(next_pre_y), np.array(ego_y))

            pso_idm = PSO(func=obj_func(args), dim=5, pop=20, max_iter=20, lb=lb_x_idm, ub=ub_x_idm, w=0.8,
                          c1=0.5, c2=0.5)
            best_x_pso_step, best_y_pso_step = pso_idm.run()

        best_x_pso.append(best_x_pso_step)
        best_y_pso.append(best_y_pso_step)

    best_x_pso = np.mean(best_x_pso, axis=0)
    np.savetxt('F:/train/ngsim'+str(id)+'/using_all_data_window_size_min_gap.txt', np.array(best_x_pso))

#
# path = r'F:\train\ngsim1742.0\ngsim1742.0.txt'
# train = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
# min_gap('1742.0', train)
#

def get_data():

    v_ids = []
    all_cf_datas = []

    filefoldnames = r'F:\train'
    for filefoldname in os.listdir(filefoldnames):
        filefold = os.path.join(filefoldnames, filefoldname)
        file = os.listdir(filefold)
        path = os.path.join(filefold, file[0])
        train = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')

        v_id = train.iloc[0,0]
        v_ids.append(v_id)

        all_cf_datas.append(train)

    return v_ids, all_cf_datas


if __name__ == "__main__":
    # a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    x0 = (1.0, 20, 0.5, 2, 2, 4)
    next_v = 0
    print("Start!")
    v_ids, all_cf_datas = get_data()
    print("v_ids", np.array(v_ids).shape)
    # print("all_cf_datas shape:", np.array(all_cf_datas).shape)

    for i in range(len(v_ids)):
        min_gap(v_ids[i], all_cf_datas[i])
    exit()

    # pool = Pool()
    # pool.map(min_gap, v_ids, all_cf_datas)
    # pool.close()
    # pool.join()





