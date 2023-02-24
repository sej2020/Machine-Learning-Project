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
    X = data[:,0]
    Y = data[:,1]
    k,d = np.polynomial.polynomial.Polynomial.fit(X,Y,1)
    y_hat = k*X + d
    plt.gca().set_aspect('equal')
    plt.title(f"{d:e} {k:e}")
    plt.plot(X,Y,'.')
    plt.plot(X,y_hat)
    plt.show()
    return k,d

#test extreme values
x,y = sys.float_info.max,sys.float_info.min
matrix1 = np.array([[x,y],[-x,-y]])
matrix2 = np.array([[x,x],[-x,-x]])
# print(make_line(matrix1))
# print(make_line(matrix2))

#test with shapes
print(np.array(make_data_circle(resolution = 0.2,radius = 1)))
points = np.array(make_data_circle(resolution = 0.2,radius = 1))

# def sse(array):
#   array[]
    


    
