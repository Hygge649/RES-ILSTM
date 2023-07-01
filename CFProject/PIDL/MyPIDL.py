from pyDOE import lhs
import numpy as np
import pandas as pd
import keras
from keras.layers import *
from keras import Input, Model
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, LambdaCallback
import math
from sklearn.metrics import mean_squared_error
import time
import keras.backend as K



from scipy.optimize import minimize

def bound(x):
    if x>0:
        return math.ceil(x)
    else:
        return math.ceil(x-1)

def IDM(x, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta=4):
    V_n_t, delta_V_n_t, S_n_t = x
    def desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n):
        item1 = S_jam_n
        item2 = V_n_t * desired_T_n
        item3 = (V_n_t * delta_V_n_t) / (2 * np.sqrt(a_max_n * a_comf_n))
        return item1 + max(0, item2 + item3)

    desired_S_n = desired_space_hw(S_jam_n, V_n_t, desired_T_n, delta_V_n_t, a_max_n, a_comf_n)
    a_n_t = a_max_n * (1 - (V_n_t / desired_V_n) ** beta - (desired_S_n / S_n_t) ** 2)

    return a_n_t


#boundary of parameters
lb = [0.1, 1, 0.1, 0.1, 0.1]
ub = [2.5, 40, 5, 10, 5]


#DATA-PREPARING
#observed data
path = r'G:\train_all\ngsim11.0\ngsim11.0.txt'
observed = pd.read_csv(path, delim_whitespace=True, dtype='float64',  encoding='utf-8')

observed_state = observed[['Mean_Speed', 'speed_diff', 'gap']]
observed_action = observed.Mean_Acceleration.values.reshape(-1,1)
parameter =pd.read_csv(r'G:\train\ngsim11.0\using_all_data_window_size.txt',
                       delim_whitespace=True, encoding='utf-8')
[a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n] = parameter.iloc[0,:-1]

#collocation point
v = observed.Mean_Speed.values.reshape(-1, 1)
v_diff = observed.speed_diff.values.reshape(-1, 1)
h = observed.gap.values.reshape(-1, 1)

#training set: collocation + part of observered
train_size = int(len(observed_state)*0.5)
test_size = int(len(observed_state)*0.25)

#input-->3-D
def create_dataset(dataset, look_back=50):
    data = []
    for i in range(len(dataset)- look_back):
        a = dataset[i:(i + look_back), :]
        data.append(a)
    return np.array(data)



N = train_size
lb_s = np.array([bound(min(v)), bound(min(v_diff)), bound(min(h))])
ub_s = np.array([bound(max(v)),  bound(max(v_diff)), bound(max(h))])
collocation_state = lb_s + (ub_s - lb_s) * lhs(3, N)


# Delete duplicate lines
observed_state = observed_state.values
collocation_state = collocation_state[~np.isin(collocation_state, observed_state).all(1)]


training_state = observed_state[0:train_size, :]
training_action = observed_action[0:train_size, :]

valid_state = observed_state[train_size:train_size+test_size, :]
valid_action = observed_action[train_size:train_size+test_size, :]

test_state = observed_state[train_size+test_size:, :]
test_action = observed_action[train_size+test_size:, :]


collocation_state = create_dataset(collocation_state)
training_state = create_dataset(training_state)

#prediction-only-train
'''
input: (observed_state, observed_action), (collocation_state, collocation_action)
output: predict_ observed_action,collocation_action of neural network
'''
#based -- pos --SIM-2 --calibration
collocation_action = []
for i in collocation_state:
    collocation_action.append(IDM(i, a_max_n, desired_V_n, a_comf_n, S_jam_n, desired_T_n, beta=4))
collocation_action = np.array(collocation_action).reshape(-1,1)
# collocation_data = np.concatenate((collocation_state, collocation_action), axis=1)



import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.enable_eager_execution()
tf.config.run_functions_eagerly(True)

import tensorflow_probability as tfp
from tensorflow.keras.layers import *
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
#
# class PINN(tf.keras.Model):
#     def __init__(self, O_S, O_A, C_S, C_A):
#         super(PINN, self).__init__()
#         self.O_S = O_S
#         self.O_A = O_A
#         self.C_S = C_S
#         self.C_A = C_A
#         #
#         # self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
#         #                                              log_device_placement=True))
#
#
#         self.pre_oa = self.PUNN(self.O_S)
#         self.pre_ca = self.PUNN(self.C_S)
#
#         delta = 0.7
#         self.loss = delta * mean_squared_error(self.O_A, self.pre_oa) + \
#                     (1-delta) * mean_squared_error(self.C_A, self.pre_ca)
#         # self.optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#         # self.optimizer = minimize(self.loss,
#         #                         method ='L-BFGS-B',
#         #                         options = {'maxiter': 50,
#         #                                    'maxfun': 50000,
#         #                                    'maxcor': 50,
#         #                                    'maxls': 50,
#         #                                    'ftol' : 1.0 * np.finfo(float).eps})
#
#         init = tf.compat.v1.global_variables_initializer()
#         self.sess.run(init)
#
#     # def loss(self, O_A, pre_oa, C_A, pre_ca):
#     #     delta = 0.7
#     #     loss = delta * mean_squared_error(O_A, pre_oa) + \
#     #                 (1-delta) * mean_squared_error(C_A, pre_ca)
#     #     return loss
#
#     def PUNN(self, x):
#         # x = create_dataset(x)
#         # x = LSTM(32, input_length=50, input_dim=3, activation='tanh')(x)
#         x = Dense(1)(x)
#
#         return x.eval(session=tf.compat.v1.Session())
#
#
#     def call(self,input):
#
#         return self.PUNN(input)
#
#
#     def train(self):
#
#         # loss = self.loss(self.O_A, self.pre_oa, self.C_A, self.pre_ca)
#         self.optimizer = minimize(self.sess,
#                             method ='BFGS',
#                             options={'maxiter': 50,})
#                                        # 'maxfun': 50000,
#                                        # 'maxcor': 50,
#                                        # 'maxls': 50,
#                                        # 'ftol' : 1.0 * np.finfo(float).eps})
#
#
#     def predict(self, test_state):
#         test_acc = self.PUNN(test_state)
#         return test_acc.numpy()


# model = PINN(observed_state, observed_action[50:,], collocation_state, collocation_action[50:,])



# from keras_preprocessing.sequence import pad_sequences
# c = (collocation_state.T).copy()
# collocation_state.T = pad_sequences(c, maxlen=len(training_state), value=0, padding='post')
#

input_c = Input(shape=(3, ), name='input_c')
input_o = Input(shape=(3, ), name='input_o')
# Add Masking layers to handle input sequences of different lengths
# len = len(input_o)
# input_c = tf.keras.preprocessing.sequence.pad_sequence(maxlen=len, value=0, padding='post')

lstm = LSTM(32, activation='tanh')

output_c = Dense(1,  name='output_c')(input_c)
output_o = Dense(1,  name='output_o')(input_o)

# lstm_c = lstm(create_dataset(input_c))
# lstm_o = lstm(create_dataset(input_o))
# output_c = Dense(1,  name='output_c')(lstm_c)
# output_o = Dense(1,  name='output_o')(lstm_o)

model = keras.Model(inputs=[input_c, input_o], outputs=[output_c, output_o])
# model = keras.Model(inputs=[input_c, input_o], outputs=[o_c, o_o])
model.summary()
print(type(model.output[0]))

# def loss(C_A, O_A, pre_ca, pre_oa):
#     delta = 0.7
#     loss = delta * mean_squared_error(O_A, pre_oa) + \
#                 (1-delta) * mean_squared_error(C_A, pre_ca)
#
#     # loss = delta * tf.reduce_mean(tf.square(O_A - pre_oa)) + (1-delta) * tf.reduce_mean(tf.square(C_A - pre_ca))
#     return loss

model.compile(optimizer="Adam",
              loss={'output_c': mean_squared_error,
                    'output_o': mean_squared_error},
              loss_weights=[0.7, 0.3],
              metrics=["accuracy"])

early_stopping = EarlyStopping(
    monitor='val_loss',
    # monitor='loss',
    patience=10,
    restore_best_weights=True,
)


history = model.fit({'input_c': collocation_state, 'input_o': training_state},
                    {'output_c': collocation_action, 'output_o': training_action},
                    epochs=5,
                    validation_data=({'input_o': valid_state},
                    {'output_o': valid_action}),
                    callbacks=early_stopping,
                    verbose=2, shuffle=False)


# model.fit({'input_c': collocation_state, 'input_o': observed_state},
#           {'output_c': collocation_action, 'output_o': observed_action},
#           epochs=5, verbose=2)

# model = PINN(observed_state, observed_action, collocation_state, collocation_action)


start_time = time.time()
model.train()
elapsed = time.time() - start_time
print('Training time: %.4f' % elapsed)

pre_acc = model.predict(observed_state)
error_u = np.linalg.norm(pre_acc - observed_action[50:,], 2) / np.linalg.norm(observed_action[50:, ], 2)
print('Error u: %e' % error_u)





