import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy as sp

def gen_hyp_ellip(axes: list, resolution: float, seed: int = 100):
    rng = np.random.default_rng(seed)
    print(axes)
    norm_points = np.array([rng.normal(loc=0.0, scale=ax, size=int(1/resolution)) for ax in axes])
    numer = np.square(norm_points)
    denom = np.expand_dims(np.square(np.array(axes)), axis=1)
    radicand = np.sum(np.divide(numer, denom), axis=0)
    d = np.sqrt(radicand)
    points = np.divide(norm_points, d)
    points_org = points.T

    return points_org

def make_data_ellipse(axes, resolution, data_remove=0.0, ran_seed=100):
    print('workin')
    a,b = axes
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
    rng = np.random.default_rng(ran_seed)
    rows_to_be_removed = rng.choice(range(len(pairs)), int(data_remove*len(pairs)), replace=False)
    pairs = np.delete(pairs, rows_to_be_removed, axis=0)
    return np.array(pairs)



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

def rotate2d(pairs, degrees):
    rot_mat = np.identity(pairs.shape[1])
    theta = np.deg2rad(degrees)
    rot_mat[0][0], rot_mat[0][1], rot_mat[1][0], rot_mat[1][1] = math.cos(theta), -math.sin(theta), math.sin(theta), math.cos(theta)
    new_pairs = np.array(rot_mat) @ np.array(pairs).T
    return new_pairs.T


def make_line(data, dim):
    fig = plt.figure()
    X = data[:,0]
    Y = data[:,1]
    if dim ==2:
        ax = fig.add_subplot()
        ax.plot(X,Y,'.')

    elif dim ==3:
        ax = fig.add_subplot(projection=f'3d')
        Z = data[:,2]
        ax.plot(X,Y,Z,'.')
        ax.set_zlabel('z')

    else:
        print("too many dims on the dancefloor")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.set_aspect('equal')
    if dim == 2:
        plt.xlim(-20, 20)
        plt.ylim(-20, 20)
    elif dim == 3:
        ax.axes.set_xlim3d(left=-20, right=20) 
        ax.axes.set_ylim3d(bottom=-20, top=20) 
        ax.axes.set_zlim3d(bottom=-20, top=20) 
    plt.show()



# def make_data_gen(lower: int, upper: int, dimensions: int, resolution: float, rot: bool):

#     axes = [random.randrange(lower, upper, 1) for i in range(dimensions)]

#     data = gen_hyp_ellip(axes,resolution)
#     data[:,[2,-1]] = data[:,[-1,2]]
#     df = pd.DataFrame(data)
#     # make_line(data)
#     df.to_csv(f"BetaDataExper/HyperEllipsoid/data/hyperell_{dimensions}-dim_3drot_0.csv", index=False, header=False)

#     if rot:
#         data[:,[2,-1]] = data[:,[-1,2]]
#         for deg in [5,15,30,90]:
#             data_rot = rotate3d(data, deg, deg, deg)
#             data_rot[:,[2,-1]] = data_rot[:,[-1,2]]
#             df = pd.DataFrame(data_rot)
#             # make_line(data_rot)
#             df.to_csv(f"BetaDataExper/HyperEllipsoid/data/hyperell_{dimensions}-dim_3drot_{deg}.csv", index=False, header=False)
#     pass


# lower = 100
# upper = 1000
# resolution = 0.001
# dimensions =  [3, 5, 10, 50, 100, 250, 500, 1000, 5000, 10000] #50,000, 100,000 ?

# for dim in dimensions:
#     make_data_gen(lower, upper, dim, resolution, rot=True)

def make_exp_data_2d(location, axes, rotation, resolution, data_remove):

    data = make_data_ellipse(axes, resolution, data_remove)
    data_rot = rotate2d(data, rotation)
    data_rot[:,0] += location[0]
    data_rot[:,1] += location[1]

    df = pd.DataFrame(data_rot)
    pass
    # df.to_csv(f"BetaDataExper/HyperEllipsoid/data/hyperell_loc-{location}_ax-{axes}_rot-{rotation}_.csv", index=False, header=False)


def make_exp_data_3d(location, axes, rotation, resolution):

    data = gen_hyp_ellip(axes,resolution)
    data_rot = rotate3d(data, rotation, rotation, rotation)
    data_rot[:,0] += location[0]
    data_rot[:,1] += location[1]
    data_rot[:,2] += location[2]

    df = pd.DataFrame(data_rot)
    make_line(data_rot, dim=3)
    df.to_csv(f"BetaDataExper/HyperEllipsoid/data/hyperell_{loc}_{axes}_{rot}_.csv", index=False, header=False)

#############################################################################################
location2d = [(x,y) for x in [0,10] for y in [0,10]]
axis_ratio2d = [[10,b] for b in range(2,12,2)]
rotations = [0, 15, 45, 60, 90]
data_remove = [0.1, 0.25, 0.5]
resolution = 0.001


for loc in location2d:
    for rot in rotations:
        for ax in axis_ratio2d:
            for dr in data_remove:

                make_exp_data_2d(location=loc, axes=ax, rotation=rot, resolution=resolution, data_remove=dr)



# make_exp_data_2d(location=(0,5.25), axes=[5,5], rotation=0, resolution=0.001)
# make_exp_data_2d(location=(0,5.025), axes=[5,5], rotation=0, resolution=0.001)
# make_exp_data_2d(location=(0,7.5), axes=[5,5], rotation=0, resolution=0.001)
##############################################################################################
# location3d = [(x,y,z) for x in [-10,0,10] for y in [-10,0,10] for z in [-10,0,10]]
# axis_ratio3d = [(10,b,c) for b in [4,10] for c in [4,10]]
# rotations = [0, 45, 90]
# resolution = 0.01

# for loc in location3d:
#     for ax in axis_ratio3d:
#         for rot in rotations:
#             make_exp_data_3d(location=loc, axes=ax, rotation=rot, resolution=resolution)
