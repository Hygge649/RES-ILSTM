import pandas as pd
import matplotlib.pyplot as plt
import os
#draw picture of each pair


def draw(ngsim_data):
    data = ngsim_data

    x1 = data["Frame_ID"].values
    x2 = data["gap"].values

    #y: follower
    y1 = data["Mean_Speed"].values
    y2 = data["LocalY"].values


    #z:leader
    z1 =  data["Mean_Speed_leader"].values
    z2 = data["LocalY_leader"].values


    plt.title("IDM_v")
    #'b', 'g', 'r', 'c', 'm', 'y', 'k'
    plt.plot(x1, y1, c='r')
    plt.plot(x1, z1, c='r',linestyle='--' )
    # plt.savefig(filefold + '/' + '/pair_v.jpg')
    plt.show()

    plt.title("IDM_x")
    #'b', 'g', 'r', 'c', 'm', 'y', 'k'
    plt.plot(x1, y2, c='g')
    plt.plot(x1, z2, c= 'g',linestyle='--' )
    # plt.savefig(filefold + '/' + '/pair_x.jpg')
    plt.show()

    # plt.title("IDM_x_v")
    # plt.plot(y2,y1,c='g')
    # plt.plot(z2,z1,c='g',linestyle='--')
    # plt.xlabel('x')
    # plt.ylabel('v')
    # # plt.savefig(filefold + '/' + 'IDM_x_v.jpg')
    # plt.savefig(filefold + '/' + '/pair_x_v.jpg')
    # plt.show()

    # plt.title("IDM_gap_v")
    # plt.plot(x2,y1,c='b')
    # plt.plot(x2,z1,c='b',linestyle='--')
    # plt.xlabel('gap')
    # plt.ylabel('v')
    # plt.savefig(filefold + '/' + '/pair_gap_v.jpg')
    # plt.show()

path =r'C:\Users\SimCCAD\Desktop\train\ngsim74.0\ngsim74.0.txt'
IDM_PATH =  r'C:\Users\SimCCAD\Desktop\train\ngsim74.0\idm_test.txt'
data = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
draw(data)



# loss_path = r'C:\Users\SimCCAD\Desktop\test_loss.csv'
# test_loss = pd.read_csv(loss_path)
# loss = test_loss[test_loss['1'] > 1]
#
#
# path = r'C:\Users\SimCCAD\Desktop\TEST'
# filefoldnames=os.listdir(path) # 930
# for filefoldname in filefoldnames:
#     filefold = 'C:\\Users\\SimCCAD\\Desktop\\TEST'+ '\\' + filefoldname
#     file = os.listdir(filefold)
#     path = filefold + '\\' + file[2]
#     # path = r'C:\Users\SimCCAD\Desktop\TEST\ngsim164.0\ngsim164.0.txt'
#     data = pd.read_csv(path, delim_whitespace=True, encoding='utf-8')
#     name = os.path.splitext(file[2])[0]
#     draw(data, filefold)
#     print(name)





