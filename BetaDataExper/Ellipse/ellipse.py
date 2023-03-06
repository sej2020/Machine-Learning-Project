import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp

def make_data_circle(radius, resolution):
    pairs = []
    for i in np.arange(-radius,radius,resolution):
        #generating angles
        alpha = 2 * math.pi * i
        # calculating coordinates
        x = radius * math.cos(alpha)
        y = radius * math.sin(alpha)
        pairs.append([x,y])
    return pairs

def make_data_ellipse(a, b, resolution):
    num = 1/resolution
    angles = 2 * np.pi * np.arange(num) / num
    e2 = (1.0 - a ** 2.0 / b ** 2.0)
    tot_size = sp.special.ellipeinc(2.0 * np.pi, e2)
    arc_size = tot_size / num
    arcs = np.arange(num) * arc_size
    res = sp.optimize.root(lambda x: (sp.special.ellipeinc(x, e2) - arcs), angles)
    angles = res.x 
    pairs = []
    for angle in angles:
        x = a * np.cos(angle)
        y = b * np.sin(angle)
        pairs.append([x,y])
    return pairs



def make_line(data):
    # print(np.size(data))
    X = data[:,0]
    Y = data[:,1]
    # k,d = np.polyfit(X,Y,1)
    # y_hat = k*X + d
    plt.gca().set_aspect('equal')
    # plt.title(f"{d:e} {k:e}")
    plt.plot(X,Y,'.')
    # plt.plot(X,y_hat)
    plt.grid()
    plt.show()
    # return k,d


#test with shapes
print(make_line(np.array(make_data_circle(radius = 1, resolution=.01))))

print(make_line(np.array(make_data_ellipse(a=4, b=1, resolution=.01))))

print(make_line(np.array(make_data_ellipse(a=2, b=1, resolution=.01))))


