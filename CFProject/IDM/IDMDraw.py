import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot(data, data_window_size):
    draw_data = pd.DataFrame(data=[data, data_window_size]).T
    draw_data.columns=['Original', 'Window_Size']
    # sns.kdeplot(data=draw_data, fill=True)
    draw_data.plot.box()
    # plt.boxplot(data=draw_data)
    # sns.violinplot(data=draw_data)
    plt.show()


a = np.array([1,2,3])
b = np.array([3,2,1])
c = [a,b]
d = np.mean(c, axis=0)


path = r'C:\Users\SimCCAD\Desktop\result\idm_error.csv'
idm_error = pd.read_csv(path)
path_1 = r'C:\Users\SimCCAD\Desktop\result\idm_error_window_size.csv'
idm_error_window_size = pd.read_csv(path_1)
print(idm_error['RMSE_GAP'].mean())
print(idm_error['RMSE_GAP'].var())
print(idm_error_window_size['RMSE_GAP'].mean())
print(idm_error_window_size['RMSE_GAP'].var())




# plot(idm_error['RMSE_V'], idm_error_window_size['RMSE_V'])


para_path = r'C:\Users\SimCCAD\Desktop\result\parameter.csv'
para = pd.read_csv(para_path)
para_path_1 = r'C:\Users\SimCCAD\Desktop\result\parameter_window_size.csv'
para = pd.read_csv(para_path)

# plot(para['Maximum_Acc'])

