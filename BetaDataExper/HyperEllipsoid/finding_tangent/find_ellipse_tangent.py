from sklearn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
# import torch
# import mxnet as mx
import pyaml
from pathlib import Path



def linreg_pipeline(X, y, location, step_n, include_regs="all", vis_theme="whitegrid", output_folder=Path("BetaDataExper/HyperEllipsoid/finding_tangent/figs/")):
    """
    Pipeline takes actual data, not a path, and runs each of the linear regression algorithms available over it
    
    Args: 

        array - array of data for circle

        include_regs - "all" to use all algorithms or a list of desired algorithms to use a subset
            options: "tf-necd" ::: "tf-cod" ::: "pytorch-qrcp" ::: "pytorch-qr" ::: "pytorch-svd" ::: "pytorch-svddc" ::: "sklearn-svddc" ::: "mxnet-svddc"
                    
        vis_theme - "darkgrid" by default, or specify any one of the below options
            options: "darkgrid" ::: "whitegrid" ::: "dark" ::: "white" ::: "ticks"
        
    Returns result_dict:

        includes various useful things 

    """
    
    reg_names = decide_regressors(include_regs)
    
    results_dict = regression_loop(X, y, reg_names)

    successful_regs = list(results_dict.keys())
     
    generate_figures(results_dict, X, y, vis_theme, successful_regs, output_folder, location, step_n)
    
    metadata = {
        "completed_regs": successful_regs,
    }
    
    dump_to_yaml(output_folder / f"metadata_loc{location}_step{step_n}.yaml", metadata, True)
    
    dump_to_yaml(output_folder / f"results_loc{location}_step{step_n}.yaml", results_dict)
    
    return results_dict
    

def decide_regressors(include_regs):
    possible_regressors = [
        "tf-necd",
        "tf-cod",
        "pytorch-qrcp",
        "pytorch-qr",
        "pytorch-svd",
        "pytorch-svddc",
        "sklearn-svddc",
        "mxnet-svddc",
    ]
    
    if include_regs == "all":
        reg_names = possible_regressors
        
    elif isinstance(include_regs, (tuple, list, set)):
        reg_names = list({reg_name for reg_name in include_regs if reg_name in possible_regressors})
        
    else: 
        raise ValueError(f'Invalid value passed for include_regs: {include_regs}\nSee documentation')
    
    return reg_names


def regression_loop(X_train, y_train, reg_names):
    results_dict = {}
        
    for reg_name in reg_names:       

        match reg_name:
            case "sklearn-svddc":
                model = linear_model.LinearRegression(fit_intercept=False).fit(X_train,y_train).coef_

            case "tf-necd":
                model = tf.linalg.lstsq(X_train, y_train[...,np.newaxis], fast=True).numpy()
                
            case "tf-cod":
                model = tf.linalg.lstsq(X_train, y_train[...,np.newaxis], fast=False).numpy()

            case "pytorch-qrcp":
                model = np.array(torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis]), driver="gelsy").solution)

            case "pytorch-qr":
                model = np.array(torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis]), driver="gels").solution)

            case "pytorch-svd":
                model = np.array(torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis]), driver="gelss").solution)

            case "pytorch-svddc":
                model = np.array(torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis]), driver="gelsd").solution)

            case "mxnet-svddc":
                model = mx.np.linalg.lstsq(X_train, y_train[...,np.newaxis], rcond=None)[0]
            

        results_dict[reg_name] = model
        
    return results_dict


def dump_to_yaml(path, object, verbose_output = True):
    print(f"Results dict: {object.keys()}")
    if not verbose_output:
        for reg in object.keys():
            del object[reg]["y_pred"]
    
    with open(path, "w") as f_log:
        dump = pyaml.dump(object)
        f_log.write(dump)


def generate_figures(results_dict, X, y, vis_theme, successful_regs, output_folder, location, step_n):

    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    CHONK_SIZE = 24
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:black")
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=CHONK_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black") #  powder blue
        
    possible_themes = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
    assert vis_theme in possible_themes, f"Invalid value passed for vis_theme: {vis_theme}\nSee documentation"
    sns.set_style(vis_theme, {'font.family':['serif'], 'axes.edgecolor':'black','ytick.left': True})
    plt.ticklabel_format(style = 'plain')

    label_lookup = {
        "tf-necd": "TensorFlow (NE-CD)",
        "tf-cod": "TensorFlow (COD)",
        "pytorch-qrcp": "PyTorch (QRCP)",
        "pytorch-qr": "PyTorch (QR)",
        "pytorch-svd": "PyTorch (SVD)",
        "pytorch-svddc": "PyTorch (SVDDC)",
        "sklearn-svddc": "scikit-learn (SVDDC)",
        "mxnet-svddc": "MXNet (SVDDC)"
    }
    
    fig, ax = plt.subplots()
    sns.scatterplot(x=X.flatten(), y=y.flatten(), ax=ax, s=3, marker=".", color="black", edgecolor="black")
    rangex = np.linspace(-50, 50, 2)[:, np.newaxis]
    reg_lines = [rangex @ results_dict[regressor] for regressor in successful_regs]
    for line, regressor in zip(reg_lines, successful_regs):
        ax.plot(rangex.flatten(), line.flatten(), label=label_lookup[regressor])
    # ax.legend()

    # plotting a thin line to accentuate x and y axes
    ax.plot([i for i in range(-50,50)], [0 for _ in range(-50,50)], linestyle="dashed", color="gray", alpha=0.4)
    ax.plot([0 for _ in range(-50,50)], [i for i in range(-50,50)], linestyle="dashed", color="gray", alpha=0.4)

    ax.set_aspect('equal')
    ax.set_xlim(-20, 20)
    ax.set_ylim(-10, 30)
    fig.suptitle(f"Circluar Data and its Regression Line")
    # ax.set_xlabel(f"")
    # ax.set_ylabel(f"")
    plt.savefig(output_folder / f"regression_line_loc{location}_step{step_n}.png")
    # plt.show()  



def main_test(path, include_regs, lr=1e-2, threshold=1e-2):
    df = pd.read_csv(path)
    location = path.split("loc-")[-1].split("_")[0]
    loc_x, loc_y = int(location.split(",")[0].strip("() ")), int(location.split(",")[1].strip("() "))
    array = df.to_numpy()
    X, y = array[:, :-1], array[:, -1]

    dist_from_regline = sys.float_info.max
    step_n = 1
    sign = 1
    while abs(dist_from_regline) > threshold:
        results = linreg_pipeline(X, y, location=location, step_n=step_n, include_regs=include_regs)
        X = X + sign*lr*abs(loc_y)
        regline = X @ results[include_regs[0]]
        min_distance = np.min(np.subtract(y, regline))
        if min_distance < 0:
            sign = -1
            lr = lr/2
        else:
            sign = 1
        dist_from_regline = min_distance
        step_n += 1
    print("Run Complete")


if __name__ == '__main__':
    path = "BetaDataExper/HyperEllipsoid/data/hyperell_loc-(0, 10)_ax-[5, 5]_rot-0_.csv"
    include_regs = ["sklearn-svddc"]
    #               "tf-necd", "tf-cod", "pytorch-qrcp", "pytorch-qr", "pytorch-svd", "pytorch-svddc", "sklearn-svddc", "mxnet-svddc"

    main_test(path, include_regs)