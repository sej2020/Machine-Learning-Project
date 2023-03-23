import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

def gen_hyp_ellip(axes: list, resolution: float, seed: int = 100):
    rng = np.random.default_rng(seed)
    norm_points = np.array([rng.normal(loc=0.0, scale=ax, size=int(1/resolution)) for ax in axes])
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

def rotate3d(pairs, yaw, pitch, roll):
    from math import cos, sin
    rot_mat = np.identity(pairs.shape[1])
    print(rot_mat)
    alpha, beta, gamma = np.deg2rad(yaw), np.deg2rad(pitch), np.deg2rad(roll)
    rotation = np.array([
        [cos(alpha)*cos(beta), cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma), cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)],
        [sin(alpha)*cos(beta), sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma), sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma)],
        [-sin(beta), cos(beta)*sin(gamma), cos(beta)*cos(gamma)]
        ])
    rot_mat[:3,:3] = rotation
    print(rot_mat)
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

lower = 100
upper = 1000
dimensions = 3
resolution = 0.001
# axes = [random.randrange(lower, upper, 1) for i in range(dimensions)]
axes = [8000, 3000, 5000]

data = gen_hyp_ellip(axes,resolution)
df = pd.DataFrame(data)

df.to_csv(f"BetaDataExper/HyperEllipsoid/data/ellipsoid_3drot_0.csv", index=False, header=False)

for deg in [5,15,30,90]:
    data_rot = rotate3d(data,deg, deg, deg)
    df = pd.DataFrame(data_rot)
    make_line(data_rot)
    df.to_csv(f"BetaDataExper/HyperEllipsoid/data/ellipsoid_3drot_{deg}.csv", index=False, header=False)