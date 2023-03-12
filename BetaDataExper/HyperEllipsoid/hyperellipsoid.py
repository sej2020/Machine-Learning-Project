import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import random

def gen_hyp_ellip(axes: list, resolution: float):
    norm_points = np.array([np.random.normal(loc=0.0, scale=ax, size=int(1/resolution)) for ax in axes])
    numer = np.square(norm_points)
    denom = np.expand_dims(np.square(np.array(axes)), axis=1)
    radicand = np.sum(np.divide(numer, denom), axis=0)
    d = np.sqrt(radicand)
    points = np.divide(norm_points, d)
    points_org = points.T

    return points_org


def rotate(pairs, degrees):
    rot_mat = np.identity(pairs.shape[1])
    theta = np.deg2rad(degrees)
    rot_mat[0][0], rot_mat[0][1], rot_mat[1][0], rot_mat[1][1] = math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)
    new_pairs = np.array(rot_mat) @ np.array(pairs).T
    return new_pairs.T


def make_line(data):
    # print(np.size(data))
    X = data[:,0]
    Y = data[:,1]
    Z = data[:,2]
    # k,d = np.polyfit(X,Y,1)
    # y_hat = k*X + d
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plt.title(f"{d:e} {k:e}")
    ax.plot(X,Y,Z,'.')
    # plt.plot(X,y_hat)
    ax.grid()
    ax.set_aspect('equal')
    plt.show()
    # return k,d

# print(np.random.normal(loc=0.0, scale=2, size = 20))
# print(make_line(gen_hyp_ellip([3,2,1],.001)))
# print(make_line(rotate(gen_hyp_ellip([3,2,1],.001),30)))

axes = [random.randrange(1, 10000, 1) for i in range(10000)]

DF = pd.DataFrame(gen_hyp_ellip(axes,.001))
DF.to_csv(f"BetaDataExper/HyperEllipsoid/data/big_hyper_ellipse.csv")

for deg in [5,15,30,90]:
    DF = pd.DataFrame(rotate(gen_hyp_ellip(axes,.001),deg))
    DF.to_csv(f"BetaDataExper/HyperEllipsoid/data/big_hyper_ellipse_rot_{deg}.csv")
