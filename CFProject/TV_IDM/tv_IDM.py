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

'''
# D(x)
    Maximum_Acc = x1  # 最大加速度
    Comfortable_Dec = x2  # 舒适减速度
    Desire_Spe = x3  # 期望速度
    Desire_Spa_Tim = x4  # 期望车头时距
    Minimum_Spa = x5  # 最短车头间距
    Para_Beta = 4  # 加速度系数. fixed as 4
    
lb_x_idm = [0.500, 0.5, 21.7, 0.1, 5]
ub_x_idm = [4.500, 4.5, 30.7, 2, 10]
'''

def set_cons(a_max_n_boundary=[0.1, 2.5], desired_V_n_boundary=[1, 40], a_comf_n_boundary=[0.1, 5],
             S_jam_boundary=[0.1, 10], desired_T_n_boundary=[0.1, 5],):# beta_boundary=[4, 4]):
#
# def set_cons(a_max_n_boundary=[0.5, 4.5], desired_V_n_boundary=[21.7, 30.7], a_comf_n_boundary=[0.5, 4.5],
#              S_jam_boundary=[0.1, 2], desired_T_n_boundary=[5,10], beta_boundary=[4, 4]):
    # constraints: eq or ineq
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


# 标准分布 生成参数
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


def tv_IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta=4):
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        return item1 + max(0, item2 + item3)

    a_n_t = []
    for i in range(len(a_max_n)):
        desired_S_n = desired_space_hw(S_jam_n[i], V_n_t, desired_T_n[i], delta_V_n_t, a_max_n[i], a_comf_n[i])
        a_n_t.append(a_max_n[i] * (1 - (V_n_t / desired_V_n[i]) ** beta - (desired_S_n / S_n_t) ** 2))

    return np.array(a_n_t)


def IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta=4):
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        return item1 + max(0, item2 + item3)

    desired_S_n = desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n)
    a_n_t = a_max_n * (1 - (V_n_t / desired_V_n) ** beta - (desired_S_n / S_n_t) ** 2)

    return a_n_t


# def sim_new_spacing(new_pre_x, new_pre_y, old_ego_x, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
#     return np.sqrt((new_pre_x - old_ego_x) ** 2 + (new_pre_y - old_ego_y) ** 2) - (
#                 2 * V_n_t + a_n_t * delta_t) / 2 * delta_t

def sim_new_spacing(new_pre_y, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
    return new_pre_y - old_ego_y - (2 * V_n_t + a_n_t * delta_t) / 2 * delta_t

def sim_new_spacing_l(new_pre_y, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
    res_spacing = []
    for i in range(len(new_pre_y)):
        res_spacing.append(new_pre_y[i] - old_ego_y[i] -
                           (2 * V_n_t[i] + a_n_t[i] * delta_t) / 2 * delta_t)
    return res_spacing


def tv_sim_new_spacing(new_pre_y, old_ego_y, V_n_t, a_n_t, delta_t=0.04):
    res_spacing = []
    for i in range(len(a_n_t)):
        res_spacing.append(new_pre_y - old_ego_y -
                           (2 * V_n_t + a_n_t[i] * delta_t) / 2 * delta_t)
    return np.array(res_spacing)




def obj_func(args):  # mse_gap
    S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_y, ego_y = args
    err = lambda x: mean_squared_error(S_n_t_1, sim_new_spacing_l(next_pre_y,  ego_y, V_n_t,
                                                                  IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1],
                                                                               x[2], x[3], x[4])))
    return err

import math
def all_rmse_using_fixed_params(args, x):
    S_n_t_1, delta_V_n_t, S_n_t, V_n_t, next_pre_y, ego_y = args
    err = mean_squared_error(S_n_t_1, sim_new_spacing_l(next_pre_y, ego_y, V_n_t,
                                                        IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, x[0], x[1], x[2], x[3], x[4], )))
    return math.sqrt(err)


def mse_using_fixed_params(args, params):  # first calculate mse_gap, then get rmse
    S_n_t_1, delta_V_n_t, S_n_t, V_n_t,  next_pre_y, ego_y = args
    err = (S_n_t_1 - sim_new_spacing(next_pre_y,  ego_y, V_n_t,
                                    IDM_cf_model_for_p(delta_V_n_t, S_n_t, V_n_t, params[0], params[1], params[2],
                                                    params[3], params[4],))) ** 2

    return err


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder " + path + " ...  ---")
        print("---  OK  ---")


# def _timed_run(func, distribution, args=(), kwargs={}, default=None):
#     """This function will spawn a thread and run the given function
#     using the args, kwargs and return the given default value if the
#     timeout is exceeded.
#     http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
#     """
#
#     class InterruptableThread(threading.Thread):
#         def __init__(self):
#             threading.Thread.__init__(self)
#             self.result = default
#             self.exc_info = (None, None, None)
#
#         def run(self):
#             try:
#                 self.result = func(args, **kwargs)
#             except Exception as err:  # pragma: no cover
#                 self.exc_info = sys.exc_info()
#
#         def suicide(self):  # pragma: no cover
#             raise RuntimeError('Stop has been called')
#
#     it = InterruptableThread()
#     it.start()
#     started_at = datetime.now()
#     it.join(self.timeout)
#     ended_at = datetime.now()
#     diff = ended_at - started_at
#
#     if it.exc_info[0] is not None:  # pragma: no cover ;  if there were any exceptions
#         a, b, c = it.exc_info
#         raise Exception(a, b, c)  # communicate that to caller
#
#     if it.isAlive():  # pragma: no cover
#         it.suicide()
#         raise RuntimeError
#     else:
#         return it.result

#
# input_data = [n*5]
# return the top 3 para. fitted distribution
def fit_posterior(data, Nbest=3, timeout=10):
    param_names = ["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]
    common_distributions = ['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm',
                            'norm', 'powerlaw', 'rayleigh', 'uniform']
    global distributions
    distributions = {}
    data = np.array(data).T
    for i in range(len(data)):
        fitted_param = {}
        fitted_pdf = {}
        sumsquare_error = {}
        y, x = np.histogram(data[i], bins=100, density=True)
        x = [(this + x[i + 1]) / 2. for i, this in enumerate(x[0:-1])]
        for distribution in common_distributions:
            try:
                # need a subprocess to check time it takes. If too long, skip it
                dist = eval("scipy.stats." + distribution)
                # dist = scipy.stats.uniform.fit()
                param = dist.fit(data[i])

                pdf_fitted = dist.pdf(x, *param)  # 概率密度函数

                fitted_param[distribution] = param[:]
                fitted_pdf[distribution] = pdf_fitted

                # calculate error
                sq_error = pylab.sum((fitted_pdf[distribution] - y) ** 2)
                sumsquare_error[distribution] = sq_error

                # calcualte information criteria
                # logLik = np.sum(dist.logpdf(x, *param))
                # k = len(param[:])
                # n = len(data[i])
                # aic = 2 * k - 2 * logLik
                # bic = n * np.log(sq_error / n) + k * np.log(n)

                # calcualte kullback leibler divergence
                # kullback_leibler = kl_div(fitted_pdf[distribution], self.y)

                # compute some errors now
                # _fitted_errors[distribution] = sq_error
                # _aic[distribution] = aic
                # _bic[distribution] = bic
                # _kldiv[distribution] = kullback_leibler
            except Exception:  # pragma: no cover
                # print("SKIPPED {} distribution (taking more than {} seconds)".format(distribution, timeout))
                # print(Exception)
                # if we cannot compute the error, set it to large values
                fitted_param[distribution] = []
                fitted_pdf[distribution] = np.nan
                sumsquare_error[distribution] = np.inf
        srt_sq_error = sorted(sumsquare_error.items(), key=lambda kv: (kv[1], kv[0]))
        for j in range(Nbest):
            dist_name = srt_sq_error[j][0]
            sq_error = srt_sq_error[j][1]
            param = fitted_param[dist_name]
            pdf = fitted_pdf[dist_name]
            if not param_names[i] in distributions:
                distributions[param_names[i]] = [
                    {"distribution": dist_name, "fitted_param": param, "sq_error": sq_error}]
            else:
                distributions[param_names[i]].append(
                    {"distribution": dist_name, "fitted_param": param, "sq_error": sq_error})
    return distributions


def initialize_params(a_max, desired_V, a_comf, S_jam, desired_T, scale=1, size=5000):
    a_max_n_boundary = [0.1, 2.5]
    desired_V_n_boundary = [1, 40]
    a_comf_n_boundary = [0.1, 5]
    S_jam_boundary = [0.1, 10]
    desired_T_n_boundary = [0.1, 5]

    def generate_one_sample(mu, sigma, boundary):
        while True:
            p = np.random.normal(mu, sigma, 1)[0]  # 生成正态分布
            if boundary[0] < p < boundary[1]:
                break
        return p

    new_params = []
    for _ in range(size):
        new_a_max = generate_one_sample(a_max, scale, a_max_n_boundary)
        new_desired_V = generate_one_sample(desired_V, scale * 5, desired_V_n_boundary)
        new_a_comf = generate_one_sample(a_comf, scale, a_comf_n_boundary)
        new_S_jam = generate_one_sample(S_jam, scale, S_jam_boundary)
        new_desired_T = generate_one_sample(desired_T, scale, desired_T_n_boundary)
        new_params.append([new_a_max, new_desired_V, new_a_comf, new_S_jam, new_desired_T])

    return np.array(new_params)


def generate_uniform_params(size=5000):
    a_max_n_boundary = [0.1, 2.5]
    desired_V_n_boundary = [1, 40]
    a_comf_n_boundary = [0.1, 5]
    S_jam_boundary = [0.1, 10]
    desired_T_n_boundary = [0.1, 5]

    new_a_max = np.random.uniform(a_max_n_boundary[0], a_max_n_boundary[1], size)
    new_desired_V = np.random.uniform(desired_V_n_boundary[0], desired_V_n_boundary[1], size)
    new_a_comf = np.random.uniform(a_comf_n_boundary[0], a_comf_n_boundary[1], size)
    new_S_jam = np.random.uniform(S_jam_boundary[0], S_jam_boundary[1], size)
    new_desired_T = np.random.uniform(desired_T_n_boundary[0], desired_T_n_boundary[1], size)
    new_params = np.array([new_a_max, new_desired_V, new_a_comf, new_S_jam, new_desired_T]).T

    return new_params


def generate_new_params(distributions, size=5000):
    a_max_n_boundary = [0.1, 2.5]
    desired_V_n_boundary = [1, 40]
    a_comf_n_boundary = [0.1, 5]
    S_jam_boundary = [0.1, 10]
    desired_T_n_boundary = [0.1, 5]


    best_posterior = {}
    for param_name in ["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]:
        best_posterior[param_name] = {"name": distributions[param_name][0]["distribution"],
                                      "param": distributions[param_name][0]["fitted_param"]}

    def generate_one_sample(best_posterior, boundary):
        while True:
            p = eval("scipy.stats." + best_posterior["name"] + ".rvs")(*best_posterior["param"], size=1)[0]
            if boundary[0] < p < boundary[1]:
                break
        return p

    new_params = []
    for _ in range(size):
        new_a_max = generate_one_sample(best_posterior["a_max"], a_max_n_boundary)
        new_desired_V = generate_one_sample(best_posterior["desired_V"], desired_V_n_boundary)
        new_a_comf = generate_one_sample(best_posterior["a_comf"], a_comf_n_boundary)
        new_S_jam = generate_one_sample(best_posterior["S_jam"], S_jam_boundary)
        new_desired_T = generate_one_sample(best_posterior["desired_T"], desired_T_n_boundary)
        new_params.append([new_a_max, new_desired_V, new_a_comf, new_S_jam, new_desired_T])

    return np.array(new_params)


import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")

def cal_tv_params(next_v, v_id, all_cf_data):  # all_cf_data of one driver
    print("-------------------------------------------------------------------------------------------------")
    print(str(next_v) + 'th vehicle with id ' + str(v_id))
    # data_array = np.array([new_delta_V_n_t, new_S_n_t, new_V_n_t, new_a, new_S_n_t_y, new_ego_x, new_ego_y, new_next_pre_x, new_next_pre_y, new_frame_id]).T
    cons = set_cons()

    data_array = all_cf_data

    #data = [delta_V_n_t, S_n_t, V_n_t, a, S_n_t_y, ego_y, next_pre_y, frame_id]
    # S_n_t_1, delta_V_n_t, S_n_t, V_n_t,  next_pre_y, ego_y = args
    delta_V_n_t = all_cf_data[:, 0]
    S_n_t = all_cf_data[:, 1]
    V_n_t = all_cf_data[:, 2]
    a = all_cf_data[:, 3]
    S_n_t_y = all_cf_data[:, 4]
    ego_y = all_cf_data[:, 5]
    next_pre_y = all_cf_data[:, 6]
    frame_id = all_cf_data[:, 7]

    args = (np.array(S_n_t_y), np.array(delta_V_n_t), np.array(S_n_t),
            np.array(V_n_t), np.array(next_pre_y), np.array(ego_y))

    # S_n_t, S_n_t_next_time
    print("spacing", np.mean(S_n_t_y), np.mean(S_n_t), np.mean(np.array(S_n_t_y) - np.array(S_n_t)))
    print(data_array.shape)

    '''
    1. original IDM
    scipy.optimize.minimize(fun, x0, args=(), method=None, jac=None, hess=None, 
    hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None)
    fun 是最优值。
    x 是最优解。
    success 表示求解器是否成功退出。
    message 描述了求解器退出的原因

    return using_all_data.txt(fixed_para)  / from result from the PSO() in IDM.py
    '''
    # path = r'C:\Users\SimCCAD\Desktop\result\parameter.csv'
    # para = pd.read_csv(path)
    # # path_1 = r'C:\Users\SimCCAD\Desktop\result\parameter_window_size.csv'
    # # idm_error_window_size = pd.read_csv(path_1)
    #
    #
    # # path = 'C:/Users/SimCCAD/Desktop/train/' + 'ngsim' + str(v_id) + '/using_all_data.txt'
    # # path = 'C:/Users/SimCCAD/Desktop/train/' + 'ngsim' + str(v_id) + '/using_all_data_window_size.txt'
    # # res_param = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
    # res_param = np.array(para.iloc[next_v-1, 1:-1]).T
    # rmse_using_all_data = all_rmse_using_fixed_params(args, res_param)


    if os.path.exists('G:/train/ngsim'+str(v_id)+'/using_all_data_min.txt'):
        res_param = np.loadtxt('G:/train/ngsim'+str(v_id)+'/using_all_data_min.txt')
        rmse_using_all_data = all_rmse_using_fixed_params(args, res_param)
    else:
        while True:
            try:
                '''
                D(X) generate a start point, then minimize the rmse_a, find the best x for IDM 
                '''
                x0 = np.asarray(initialize())
                res = minimize(obj_func(args), x0, constraints=cons, method='trust-constr')
                if res.success:
                    break
            except ValueError:
                continue
        rmse_using_all_data = res.fun
        # mkdir('F:/train/ngsim'+str(v_id)+'/')
        # mkdir('F:/train/ngsim' + str(v_id) + '/posterior_figure/')
        np.savetxt('G:/train/ngsim'+str(v_id)+'/using_all_data_min.txt', np.array(res.x))


        res_param = res.x


    fix_a_max = res_param[0]
    fix_desired_V = res_param[1]
    fix_a_comf = res_param[2]
    fix_S_jam = res_param[3]
    fix_desired_T = res_param[4]
    fix_beta = 4

    '''
    2. IDM with time-varying:
        define the marginal prior distribution form of each para.    fixed-para. set

    '''

    # 1.initialize
    a_max = res_param[0]
    desired_V = res_param[1]
    a_comf = res_param[2]
    S_jam = res_param[3]
    desired_T = res_param[4]
    beta = 4

    # 2.hyper-para.
    sample_size = 2000
    i = 0
    fix_sum_err = 0
    tv_sum_err = 0

    tv_params = []
    # for t in T: each para at each time step ia modeled as a distribution
    for frame in data_array:
        print(str(next_v) + "th vehicle v_id " + str(v_id) + " frame " + str(i))

        para = []

        delta_V_n_t = frame[0]
        S_n_t = frame[1]
        V_n_t = frame[2]
        a = frame[3]
        S_n_t_y = frame[4]
        ego_y = frame[5]
        next_pre_y = frame[6]
        frame_id = frame[7]

        args = (S_n_t_y, delta_V_n_t, S_n_t, V_n_t, next_pre_y, ego_y)

        fix_err = mse_using_fixed_params(args, res_param)  # for original IDM

        # if fix_err < 0.001:
        #     accept_threshold = fix_err
        # else:
        #     accept_threshold = fix_err / 2

        # if fix_err < 30:
        #     accept_threshold = fix_err
        # else:
        #     accept_threshold = 30

        accept_threshold = fix_err

        fix_sum_err += fix_err
        scale = 1
        iters = 0
        max_iters = 200
        this_sample_size = copy.deepcopy(sample_size)
        accept_tv_params = []
        while iters < max_iters:
            if len(accept_tv_params) == 0 and iters > max_iters * 0.8:  # 迭代次数大于400时， 接受的样本数为0， 提高阈值（容错）
                accept_threshold = fix_err + 0.01

            if (iters == 0 and i == 0) or (len(accept_tv_params) < 100 and iters > 10):  # 正常
                if iters > 100 and len(accept_tv_params) == 0:  # 迭代次数大于100时， 接受的样本数为0
                    # print("uniform", len(accept_tv_params), fix_err, opt_err)
                    if fix_err < opt_err:  # 在定义域内  随机  生成5000组参数
                        new_params = initialize_params(fix_a_max, fix_desired_V, fix_a_comf, fix_S_jam, fix_desired_T,
                                                       scale=scale, size=sample_size)
                    else:  # 在定义域内按照  标准分布  生成5000组参数
                        new_params = generate_uniform_params(size=sample_size)

                # elif (iters % 10 == 0) and (len(accept_tv_params) != 0 and iters > 100):#迭代次数大于100且是10的倍数时， 接受的样本数大于0
                elif (iters % 10 == 0) or (
                        len(accept_tv_params) != 0 and iters > 100):  # 迭代次数是10的倍数  迭代次数大于100且接受的样本数大于0时，
                    if iters % 10 == 0:  # 迭代次数是10的倍数时, 在定义域内  随机  生成10000组参数
                        this_sample_size += 5000
                    # print("augment sample size", len(accept_tv_params), scale, this_sample_size, a_max)
                    new_params = initialize_params(a_max, desired_V, a_comf, S_jam, desired_T, scale=scale,
                                                   size=this_sample_size)

                else:  # 0在定义域内  随机  生成5000组参数
                    # print("do nothing", len(accept_tv_params), scale, this_sample_size, a_max)
                    new_params = initialize_params(a_max, desired_V, a_comf, S_jam, desired_T, scale=scale,
                                                   size=sample_size)

            else:
                try:
                    new_params = generate_new_params(distributions, size=sample_size)
                except:
                    new_params = initialize_params(a_max, desired_V, a_comf, S_jam, desired_T, scale=scale,
                                                   size=sample_size)

            # return new_params

            a_n_t_hat = tv_IDM_cf_model(delta_V_n_t, S_n_t, V_n_t, new_params.T[0], new_params.T[1], new_params.T[2],
                                        new_params.T[3], new_params.T[4], 4)

            S_n_t_y_hat = tv_sim_new_spacing(next_pre_y, ego_y, V_n_t, a_n_t_hat, )

            err = (S_n_t_y_hat - S_n_t_y)**2
            # err = math.sqrt(((S_n_t_y_hat - S_n_t_y)**2).mean())
            accept_idx = np.where(err < accept_threshold)[0]
            new_accept_tv_params = new_params[accept_idx]

            # return  new_accept_tv_params

            if len(accept_tv_params) > 100 or (len(accept_tv_params) == 0):
                accept_tv_params = new_accept_tv_params
            else:
                accept_tv_params = np.vstack((accept_tv_params, new_accept_tv_params))  # 按垂直方向（行顺序）堆叠数组构成一个新的数组

            sum_err = np.sum(err[accept_idx])
            opt_err = np.min(err)
            opt_err = math.sqrt(opt_err)
            print(next_v, "v_id", v_id, "frame", i, "iters", iters, "accept params num", accept_tv_params.shape,
                  "fix_err", math.sqrt(fix_err), "opt_err", opt_err)

            if len(accept_tv_params) > 100:
                distributions: Dict[str, List[Dict[str, Union[str, Any]]]] = fit_posterior(accept_tv_params)
                if len(accept_tv_params) > sample_size * 0.95:
                    break
            else:
                # a_max, desired_V, a_comf, S_jam, desired_T
                new_mean = new_params[np.argmin(err)]
                a_max = new_mean[0]
                desired_V = new_mean[1]
                a_comf = new_mean[2]
                S_jam = new_mean[3]
                desired_T = new_mean[4]
            iters += 1

            para.append(new_params[np.argmin(err)])
            # print(np.array(tv_params))
        # return best_para for each frame, amd use the pare_set to sim the trajectorty

        tv_params.append(np.mean(para,  axis=0))

        print(next_v, "v_id", v_id, "frame", i, "tv_err", math.sqrt(sum_err / len(accept_tv_params)), "opt_err", opt_err,
              "fix_err", math.sqrt(fix_err))


        # print(next_v, "v_id", v_id, "frame", i, 'tv_para', tv_params)
        # if len(accept_tv_params) > sample_size * 0.95:
        #     np.savetxt('C:/Users/SimCCAD/Desktop/data/0414_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt',
        #                 np.array(accept_tv_params))
        # elif len(accept_tv_params) != 0:
        #     np.savetxt('C:/Users/SimCCAD/Desktop/data/0414_mop_space_dist_param/' + str(int(v_id)) + '/' + str(int(i)) + '_tv_params.txt',
        #                np.array(accept_tv_params))

        for param_name in ["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]:
            # plt.figure()
            # distributions[param_name].summary()
            # plt.savefig('0714_dist_param/' + str(int(v_id)) + '/posterior_figure/' + param_name + '_' + str(int(i)) + '.png')
            print("--------------------------------------------------------------------------------------------")
            for dist in distributions[param_name]:
                print(v_id, param_name, dist["distribution"], dist["fitted_param"], dist["sq_error"])

        tv_sum_err += sum_err / len(accept_tv_params)
        # tv_params.append(accept_tv_params)
        i += 1
    # print(tv_params)
    # parameter_path = r'C:\Users\SimCCAD\Desktop\result\tv_params.txt'
    # pd.DataFrame(np.array(tv_params)).to_csv(parameter_path, sep='\t', index=False)


    np.savetxt('G:/train/ngsim' + str(v_id) + '/tv_params.txt', np.array(tv_params))
    print("all data %d | RMSE: %.4f | a_max: %.4f | desired_V: %.4f | a_comf: %.4f | S_jam: %.4f | desired_T: %.4f | beta: %.3f" % \
        (v_id, rmse_using_all_data, res_param[0], res_param[1], res_param[2], res_param[3], res_param[4], res_param[5]))
    print(str(int(v_id)), "RMSE:", np.sqrt(fix_sum_err / len(a)), np.sqrt(tv_sum_err / len(a)))
    # print(str(int(v_id)), "mean:", np.mean(np.abs(a-a_hat)), np.std(np.abs(a-a_hat)))
    # tv_params = np.array(tv_params)
    # print(str(int(v_id)), "tv params:", tv_params.shape)

    # f = open('0623_res_tv_desired_v/prior' + str(int(v_id)) + '.pkl', 'wb')
    # pickle.dump(prior, f)
    # f.close()

import pandas as pd

def get_data_with_pos():

    next_vs = []
    v_ids = []
    all_cf_datas = []
    next_v = 1

    '''
    trainform train = ['Vehicle_ID','Leader_ID',  "Frame_ID",  "Mean_Speed", 'Mean_Speed_leader', 'LocalY',  'LocalY_leader',
     'Mean_Acceleration','Vehicle_length', 'Space_Headway', "speed_diff", "gap"]

     to data_array 
    '''

    filefoldnames = r'G:\train'
    for filefoldname in os.listdir(filefoldnames):
        filefold = os.path.join(filefoldnames, filefoldname)
        file = os.listdir(filefold)
        path = os.path.join(filefold, file[0])
        train = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')

        next_vs.append(next_v)
        v_id = train.iloc[0,0]
        v_ids.append(v_id)

        print("-------------------------------------------------------------------------------------------------")
        print(str(next_v) + 'th vehicle with id ' + str(v_id))

        next_v += 1

        delta_V_n_t = train.speed_diff[:-1]
        S_n_t = train.gap[:-1]
        V_n_t = train.Mean_Speed[:-1]
        a = train.Mean_Acceleration[:-1]
        ego_y = train.LocalY[:-1]

        S_n_t_y = train.gap[1:]
        next_pre_y = train.LocalY_leader[1:]
        frame_id = train.Frame_ID[:-1]

        data_array = np.array([delta_V_n_t, S_n_t, V_n_t, a, S_n_t_y, ego_y, next_pre_y, frame_id]).T
        all_cf_datas.append(data_array)

    return next_vs, v_ids, all_cf_datas


if __name__ == "__main__":
    # a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta
    x0 = (1.0, 20, 0.5, 2, 2, 4)
    next_v = 0
    print("Start!")
    next_vs, v_ids, all_cf_datas = get_data_with_pos()
    # print("next_vs", np.array(next_vs).shape)
    # print("v_ids", np.array(v_ids).shape)
    # print("all_cf_datas shape:", np.array(all_cf_datas).shape)

    # for i in range(len(next_vs)):
    #     cal_tv_params(next_vs[i], v_ids[i], all_cf_datas[i])
    # exit()

    pool = Pool()
    pool.map(cal_tv_params, next_vs, v_ids, all_cf_datas)
    pool.close()
    pool.join()