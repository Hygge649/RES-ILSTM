import pandas as pd
import os

filefoldnames = r'C:\Users\SimCCAD\Desktop\NGSIM_1min' #222

train_size = int(len(os.listdir(filefoldnames)) *0.8)   #177/45
train_set = pd.DataFrame()
test_set = pd.DataFrame()
count = 0

for filefoldname in os.listdir(filefoldnames):
    filefold = os.path.join(filefoldnames, filefoldname)
    file = os.listdir(filefold)
    path = os.path.join(filefold, file[0])
    data = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')

    if count < train_size:#data is train_data
        train_set = pd.concat([train_set, data], axis=0)

    else:
        test_set = pd.concat([test_set, data], axis=0)

    count +=1

from pathlib import Path
datapath = Path('../data')
os.makedirs(datapath)
train_path = '../data/NGSIM_train'
train_set.to_csv(train_path, sep='\t', index=False)
test_path = '../data/NGSIM_test'
test_set.to_csv(test_path, sep='\t', index=False)