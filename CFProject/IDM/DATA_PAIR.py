import pandas as pd
import os

#depart dataset into CF pairs

path = r'C:\Users\SimCCAD\Desktop\pair.txt'
data = pd.read_csv(path, names=['Vehicle_ID', 'Leader_ID', 'Frame_ID', 'Mean_Speed', 'Mean_Speed_leader',
                              'LocalY','LocalY_leader', 'Mean_Acceleration','Vehicle_length','Space_Headway',
                              'speed_diff','gap'], delim_whitespace= True, header=None,encoding='utf-8')


data['frame_diff'] = data.groupby('Vehicle_ID')['Frame_ID'].diff()-1

#1.groupby
groupby_data = data.groupby(['Vehicle_ID', 'Leader_ID', 'frame_diff'])

#2.turn to list
#groups =data.apply(lambda x:x.values.tolist())
#groups = result_groupby[:10]  #10 pair
result_groupby=groupby_data.apply((lambda x:x.values.tolist()))  #2089

#3.首先转化为字典，字典的键为groupby分组的索引，值为被分组列聚合来的list
result_dict=dict(result_groupby)
c = 0
# res_filefold = r'C:\Users\SimCCAD\Desktop\RENGSIM_1min'
for value in result_dict.values():
    value = pd.DataFrame(value, columns=['Vehicle_ID','Leader_ID', 'Frame_ID', 'Mean_Speed', 'Mean_Speed_leader',
                              'LocalY','LocalY_leader', 'Mean_Acceleration','Vehicle_length','Space_Headway',
                              'speed_diff','gap','frame_diff'])
    value = value.iloc[:, :-1]
    i = value.iloc[0][0]

    if len(value) > 600: #222
        c +=1
        res_filefold = r'C:\Users\SimCCAD\Desktop\NGSIM_1min' + r'\ngsim' + str(i) + '\\'
        os.makedirs(res_filefold)
        res_file = res_filefold + r'\ngsim' + str(i) + '.txt'
        value.to_csv(res_file, sep='\t', index=False)


    # if len(value)>500: #
    #     res_file = res_filefold + r'\ngsim' + str(i) + '.txt'
    #     value.to_csv(res_file, sep='\t', index=False)
