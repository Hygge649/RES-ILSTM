import os
import pandas as pd
import numpy as np
import scipy
import pylab
from matplotlib import pyplot as plt


path = r'G:\result\parameter_window_size.csv'
paras = pd.read_csv(path, sep=',', index_col=0)

from fitter import Fitter
data = paras[['Maximum_Acc', 'Desire_Spa_Tim']]

f = Fitter(paras.Desire_Spa_Tim, distributions=['cauchy', 'chi2', 'expon', 'exponpow', 'gamma', 'lognorm',
                            'norm', 'powerlaw', 'rayleigh', 'uniform'], timeout =100)  # 创建Fitter类
#defult: bins=100
f.fit()  # 调用fit函数拟合分布
f.plot_pdf(Nbest=3, lw=2, method='sumsquare_error')
f.summary(plot=True)
print(f.get_best())
plt.show()
dis = f.get_best().keys()
print(f.fitted_pdf)#all fitted distributions

