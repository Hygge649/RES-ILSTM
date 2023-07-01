import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pyDOE import lhs

def bound(x):
    if x>0:
        return math.ceil(x)
    else:
        return math.ceil(x-1)




x = [0,1,2]
y = [0,1]
'''
meshgrid: generate Coordinate matrix -- 
        Each element in the x-coordinate matrix X 
        and the corresponding position element in the y-coordinate matrix Y 
        together form the complete coordinate of a point
'''
X, Y = np.meshgrid(x,y)

# A = np.array([[1,2],
#               [3,4],
#               [5,6],
#               [7,8]])
#
# B = np.array([[3,4],
#               [7,8]])
#
# C = A[~np.isin(A, B).all(1)]


path = r'F:\train--SIM1,2\ngsim11.0\ngsim11.0.txt'
observed = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
observed = observed[['Mean_Speed', 'speed_diff', 'gap']]
v = observed.Mean_Speed.values.reshape(-1, 1)
v_diff = observed.speed_diff.values.reshape(-1, 1)
h = observed.gap.values.reshape(-1, 1)

size = len(v)
N = int(size*0.8)
lb = np.array([bound(min(v)), bound(min(v_diff)), bound(min(h))])
ub = np.array([bound(max(v)),  bound(max(v_diff)), bound(max(h))])
collocation = lb + (ub - lb) * lhs(3, N)
observed = observed.values
collocation= collocation[~np.isin(collocation, observed).all(1)]  #Delete duplicate lines

ax3 = plt.axes(projection='3d')
ax3.scatter(v, v_diff, h, s=15, c='Orange', marker="o")
ax3.scatter(collocation[:,0], collocation[:,1], collocation[:,2], s=15, c='red', marker="^")
# ax3.scatter(x, y, z)#, s=20, c=d, cmap="jet", marker="o")
plt.show()
