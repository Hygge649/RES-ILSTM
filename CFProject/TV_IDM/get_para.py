#concatate all the para
import os
import pandas as pd

filefoldnames = r'G:\train'
count = len(os.listdir(filefoldnames))
parameter = pd.DataFrame(columns=["a_max", "desired_V", "a_comf", "S_jam", "desired_T"])

for filefoldname in os.listdir(filefoldnames):
    filefold = filefoldnames + '\\' + filefoldname

    file = os.listdir(filefold)
    path = os.path.join(filefold, file[4])
    para = pd.read_csv(path, header=None, delim_whitespace=True, encoding='utf-8')
    para.columns=["a_max", "desired_V", "a_comf", "S_jam", "desired_T"]
    parameter = pd.concat([parameter, para], axis=0)

para = parameter.mean()

