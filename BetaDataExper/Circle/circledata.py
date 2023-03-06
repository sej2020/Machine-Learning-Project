import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def make_data_circle(resolution,radius):
    xr = np.arange(-radius,radius,resolution)
    pairs = [[radius,0]]
    for x in xr:
        y = math.sqrt(radius**2 - x**2)
        pairs.append([x,y])
        pairs.append([x,-y])
    return pairs

def make_line(data):
    # print(np.size(data))
    X = data[:,0]
    Y = data[:,1]
    k,d = np.polyfit(X,Y,1)
    y_hat = k*X + d
    # plt.gca().set_aspect('equal')
    # plt.title(f"{d:e} {k:e}")
    # plt.plot(X,Y,'.')
    # plt.plot(X,y_hat)
    # plt.show()
    return k,d

#test extreme values
x,y = sys.float_info.max,sys.float_info.min
matrix1 = np.array([[x,y],[-x,-y]])
matrix2 = np.array([[x,x],[-x,-x]])
# print(make_line(matrix1))
# print(make_line(matrix2))

#test with shapes
print(make_line(np.array(make_data_circle(resolution = 0.0000001,radius = 1))))



# run w/ lines (no graph)

# $ "C:/Program Files (x86)/Microsoft Visual Studio/Shared/Python39_64/python.exe
# " "d:/linear regression/d1.py"
# C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python39_64\lib\site-packages\numpy\lib\polynomial.py:666: RuntimeWarning: overflow encountered in multiply
#   scale = NX.sqrt((lhs*lhs).sum(axis=0))
# d:\linear regression\d1.py:31: RankWarning: Polyfit may be poorly conditioned
#   print(make_line(matrix1))
# (0.0, 0.0)
# d:\linear regression\d1.py:32: RankWarning: Polyfit may be poorly conditioned
#   print(make_line(matrix2))
# (0.0, 0.0)

# run w/ circle
# .00000001 returns domain error