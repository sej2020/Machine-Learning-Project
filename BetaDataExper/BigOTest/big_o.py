import numpy as np
import pandas as pd
from time import perf_counter_ns, process_time_ns
from sklearn import linear_model
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch
import mxnet as mx
from pathlib import Path
import pickle
import pyaml


def get_data_array(rows, cols, seed=100, input_data_path=None, output_data=False) -> np.array:
    """
    Datapath -> np.array
    """
    if input_data_path:
        with open(Path(input_data_path), 'rb') as f:
            df = pickle.load(f)

    else:
        rng = np.random.default_rng(seed=seed)
        data = rng.normal(loc=0, scale=1, size=(rows, cols))
        df = pd.DataFrame(data)

        if output_data:
            with open(Path.cwd() / 'big_o_data.pkl', 'wb') as f:
                pickle.dump(df, f)

    print(df.head(3))
    array = df.to_numpy()
    return array


def actual_expr(X_train: np.array, y_train: np.array, timer: object, reg_names: list, rows_in_expr: list) -> dict:
    """
    This function will record the runtimes to create a model of each specified regressor using a dataset of varying size. The size of the dataset will vary according to a schedule
    specified by rows_in_expr parameter. The output will be a dictionary recording these results
    Args:
        X_train (np.array) - array of full dataset attributes
        y_train (np.array) - array of full dataset target variable
        timer (timer object) - timer either perf_counter or process time
        reg_names (list) - list of regressors that will be in experiment
        rows_in_expr (list) - a list of rows that will be used in experiment e.g. [10, 100, 1000, 10000] 

    Returns:
        results_dict (dict) - dictionary of format {regressor: [list of runtimes for each # of rows specified in rows_in_expr]}

    """
    results_dict = {}
    failed_regs = []
    exceptions_lst = []

    for reg_name in reg_names:
        print(f"starting actual experiment with {reg_name}")
         
        time_list = []
        for row_count in rows_in_expr:
            partial_X_train = X_train[:row_count, :]
            partial_y_train = y_train[:row_count] 

            try:
                match reg_name:

                    case "sklearn-svddc":
                        start_lstsq = timer()
                        model = linear_model.LinearRegression(fit_intercept=False).fit(partial_X_train, partial_y_train).coef_
                        stop_lstsq = timer()

                    case "tf-necd":
                        start_lstsq = timer()
                        model = tf.linalg.lstsq(partial_X_train, partial_y_train[...,np.newaxis], fast=True).numpy()
                        stop_lstsq = timer()

                    case "tf-cod":
                        start_lstsq = timer()
                        model = tf.linalg.lstsq(partial_X_train, partial_y_train[...,np.newaxis], fast=False).numpy()
                        stop_lstsq = timer()

                    case "pytorch-qrcp":
                        start_lstsq = timer()
                        model = np.array(torch.linalg.lstsq(torch.Tensor(partial_X_train), torch.Tensor(partial_y_train[...,np.newaxis]), driver="gelsy").solution)
                        stop_lstsq = timer()

                    case "pytorch-qr":
                        start_lstsq = timer()
                        model = np.array(torch.linalg.lstsq(torch.Tensor(partial_X_train), torch.Tensor(partial_y_train[...,np.newaxis]), driver="gels").solution)
                        stop_lstsq = timer()
                            
                    case "pytorch-svd":
                        start_lstsq = timer()
                        model = np.array(torch.linalg.lstsq(torch.Tensor(partial_X_train), torch.Tensor(partial_y_train[...,np.newaxis]), driver="gelss").solution)
                        stop_lstsq = timer()

                    case "pytorch-svddc":
                        start_lstsq = timer()
                        model = np.array(torch.linalg.lstsq(torch.Tensor(partial_X_train), torch.Tensor(partial_y_train[...,np.newaxis]), driver="gelsd").solution)
                        stop_lstsq = timer()

                    case "mxnet-svddc":
                        start_lstsq = timer()
                        model = mx.np.linalg.lstsq(partial_X_train, partial_y_train[...,np.newaxis], rcond=None)[0]
                        stop_lstsq = timer()

            except Exception as e:
                failed_regs.append(reg_name)
                exceptions_lst.append(e)

            time_list += [(stop_lstsq - start_lstsq)] 

        final = []
        for i,j in zip(rows_in_expr, time_list):
            final.append([i,j])

        results_dict[reg_name] = final   

    return results_dict, failed_regs, exceptions_lst


def set_time_type(time_type: str) -> object:
    match time_type:
        case "total":
            timer = perf_counter_ns   
        case "process":
            timer = process_time_ns      
        case _:
            raise ValueError(f"time_type must be one of the options shown in the docs, not: {time_type}")
    return timer


def comp_complexity_dict(reg: str):
    """
    Retrieves a lambda function for the theoretical number of flops for the least squares solver employed by each library
    lambda x takes an x of form (m, n, r)
    |-----------------------------------------------------------------------------------------------------|
    |   Regressor    |               Solver                 | Computational Complexity                    |
    |-----------------------------------------------------------------------------------------------------|
    |    tf-necd     |       Cholesky Decomposition         |  O(mn^2 + n^3)                              |
    |     tf-cod     |   Complete Orthogonal Decomposition  |  O(2mnr - r^2*(m + n) + 2r^3/3 + r(n - r))  |
    |  pytorch-qrcp  |    QR Factorization with Pivoting    |  O(4mnr - 2r^2*(m + n) + 4r^3/3)            |
    |   pytorch-qr   |          QR Factorization            |  O(2mn^2 - 2n^3/3)                          |
    |  pytorch-svd   |            Complete SVD              |  O(4mn^2 + 8n^3)                            |
    | pytorch-svddc  |       SVD Divide-and-Conquer         |  O(mn^2)                                    |
    | sklearn-svddc  |       SVD Divide-and-Conquer         |  O(mn^2)                                    |
    |  mxnet-svddc   |       SVD Divide-and-Conquer         |  O(mn^2)                                    |
    |-----------------------------------------------------------------------------------------------------|
    

    """
    dict = {
        "tf-necd": lambda x: math.floor(x[0]*x[1]**2 + x[1]**3),
        "tf-cod": lambda x: math.floor(2*x[0]*x[1]*x[2] - x[2]**2*(x[0] + x[1]) + 2*x[2]**3/3 + x[2]*(x[1] - x[2])),
        "pytorch-qrcp": lambda x: math.floor(4*x[0]*x[1]*x[2] - 2*x[2]**2*(x[0] + x[1]) + 4*x[2]**3/3),
        "pytorch-qr": lambda x: math.floor(2*x[0]*x[1]**2 - 2*x[1]**3/3),
        "pytorch-svd": lambda x: math.floor(4*x[0]*x[1]**2 + 8*x[1]**3),
        "pytorch-svddc": lambda x: math.floor(x[0]*x[1]**2),
        "sklearn-svddc": lambda x: math.floor(x[0]*x[1]**2),
        "mxnet-svddc": lambda x: math.floor(x[0]*x[1]**2)
        }
    
    return dict[reg]


def theoretical_expr(n: int, r: int, timer: object, reg_names: list, rows_in_expr: list) -> dict:
    """
    This function will record the runtimes to perform the theoretical number of flops for the least squares solver employed by each library for a specified
    number of rows. The rows_in_expr list contains the varying number of rows in this experiment. This will be performed for
    each regressor and the output will be a dictionary recording these results
    Args:
        n (int) - the number of columns of the dataset
        r (int) - the rank of the dataset
        timer (timer object) - timer either perf_counter or process_time
        reg_names (list) - list of regressors that will be in experiment
        rows_in_expr (list) - a list of the orders of magnitude of rows that will be used in experiment e.g. [10, 100, 1000, 10000] 

    Returns:
        results_dict (dict) - dictionary of format {regressor: [list of theoretical runtimes for each # of rows specified in rows_in_expr]}
    """

    exper_vals = [(rows,n,r) for rows in rows_in_expr]
    results_dict = {}
    for reg_name in reg_names:
        print(f"starting theoretical experiment with {reg_name}")
        func = comp_complexity_dict(reg_name)
        flops = list(map(func, exper_vals))

        final = []
        for i,j in zip(rows_in_expr, flops):
            final.append([i,j])

        results_dict[reg_name] = final

    return results_dict 


def make_viz(actual_time_dict: dict, theory_time_dict: dict, timer: object, rows_in_expr: list):
    """
    Will save visualizations for Theoretical Runtime vs. Actual Runtime comparison to user's CWD

    Args:
        actual_time_dict (dict) - results from actual runtime experiment
        theory_time_dict (dict) - results from theoretical runtime experiment
        timer (object) - timer either perf_counter or process time
        rows_in_expr (list) - a list of the orders of magnitude of rows used in experiment e.g. [10, 100, 1000, 10000] 

    Returns:
        Saves figures to user's CWD
    """

    label_dict ={
        "tf-necd": "TensorFlow (NE-CD)",
        "tf-cod": "TensorFlow (COD)",
        "pytorch-qrcp": "PyTorch (QRCP)",
        "pytorch-qr": "PyTorch (QR)",
        "pytorch-svd": "PyTorch (SVD)",
        "pytorch-svddc": "PyTorch (SVDDC)",
        "sklearn-svddc": "scikit-learn (SVDDC)",
        "mxnet-svddc": "MXNet (SVDDC)"
    }
    
    SMALL_SIZE = 10
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    CHONK_SIZE = 24

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:black")
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black") #  powder blue
        
    sns.set_style("whitegrid", {'font.family':['serif'], 'axes.edgecolor':'black','ytick.left': True})

    # for reg in actual_time_dict.keys():
    #     plt.plot(rows_in_expr, actual_time_dict[reg], label=label_dict[reg]+' Actual')
    #     plt.legend()
    #     plt.title("Theoretical Bound vs. Actual Runtime")
    #     plt.ylabel("Time in nanoseconds")
    #     plt.xlabel("Number of rows in dataset")
    #     plt.savefig(f"BetaDataExper/BigOTest/figs/bigO_{reg}")
    #     plt.clf()

    # for reg in actual_time_dict.keys():
    #     plt.plot(rows_in_expr, actual_time_dict[reg], label=label_dict[reg]+' Actual')
    #     plt.plot(rows_in_expr, theory_time_dict[reg], label=label_dict[reg]+' Theoretical')
    #     plt.legend()
    #     plt.title("Theoretical Bound vs. Actual Runtime")
    #     plt.ylabel("Time in nanoseconds")
    #     plt.xlabel("Number of rows in dataset")
    #     plt.savefig(f"BetaDataExper/BigOTest/figs/bigO_{reg}")
    #     plt.clf()

    for i, reg in enumerate(actual_time_dict.keys()):
        plt.plot(rows_in_expr, [theo / act for theo, act in zip(theory_time_dict[reg],actual_time_dict[reg])], label=label_dict[reg], color=f'C{i}')
        plt.title("Ratio between Theoretical and Actual Runtimes")
        plt.legend(loc=2)
        plt.xlabel("Number of rows in dataset")
        plt.xscale("log")
        plt.savefig(f"BetaDataExper/BigOTest/figs/bigO_ratio_{reg}")
        plt.clf()
    
    for reg in actual_time_dict.keys():
        plt.plot(rows_in_expr, [theo / act for theo, act in zip(theory_time_dict[reg],actual_time_dict[reg])], label=label_dict[reg])
        plt.title("Ratio between Theoretical and Actual Runtimes")
        plt.legend(loc=2)
        plt.xscale("log")
        plt.xlabel("Number of rows in dataset")

    plt.savefig(Path.cwd() / "figs/bigO_ratio_aggregate")


def dump_to_yaml(path: str, object: dict):
    """
    Dumps a dictionary to a yaml file
    """
    print(f"Dumping to yaml: {object.keys()}")

    with open(path, "w") as f_log:
        dump = pyaml.dump(object)
        f_log.write(dump)


def main(time_type: str, reg_names: list, data_rows: int, data_cols: int, granularity=2, repeat=10):
    """
    Makes visualizations for Theoretical Runtime vs. Actual Runtime comparison

    Args:
        time_type (str): "process" to get a time without sleep or "total" to get an actual runtime
        reg_names (list): list of the regressors to be used
        data_rows (int): dataset rows for experiment
        data_cols (int): dataset columns for experiment
        granularity (int): step size of test between orders of magnitude
        repeat (int): how many times to repeat experiment

    Returns:
        Saves figures to cwd 
            OR
        Saves results as yaml file
    """

    timer = set_time_type(time_type)
    array = get_data_array(data_rows, data_cols)

    m, n = np.shape(array)
    r = np.linalg.matrix_rank(array)

    ## loop to find the maximum number of rows allowed in experiment
    max_row_bound = 0
    for i in range(1*10, (len(str(m))+1)*10, granularity):
        if 10**(i/10) <= m:
            max_row_bound = i/10
        else:
            max_row_bound = i/10
            break

    rows_in_expr = [math.floor(10**(row_bound/10)) for row_bound in range(1*10, int(max_row_bound*10), granularity) for _ in range(repeat)] # to produce orders of magnitude experiment for _ in range(repeat)
    print(f'Rows in Experiment: {rows_in_expr}')

    X, Y = array[:,:-1], array[:,-1] 

    print('All setup')
    print('running actual experiments...')

    actual_time_dict, failed_regs, exceptions_lst = actual_expr(X, Y, timer, reg_names, rows_in_expr)

    print('All done with actual experiments')

    print('now running theoretical experiments...')

    theory_time_dict = theoretical_expr(n, r, timer, reg_names, rows_in_expr)

    print(f'Actual Time: {actual_time_dict}\n--------------\nTheoretical Time: {theory_time_dict}')

    metadata = {
        "dataset_shape": f"{data_rows} x {data_cols}",
        "failed_regs": failed_regs,
        "failed_regs_exceptions": exceptions_lst,
        "rows_in_experiment": rows_in_expr,
        "timer_method": f"{time_type} in nanoseconds"
    }

    dump_to_yaml(Path.cwd() / "metadata.yaml", metadata)
    dump_to_yaml(Path.cwd() / "theoretical_time.yaml", theory_time_dict)
    dump_to_yaml(Path.cwd() / "actual_time.yaml", actual_time_dict)

    ########## IF VISUALIZING ############
    # print('All done with theoretical experiments, now just making viz')
    # make_viz(actual_time_dict, theory_time_dict, timer, rows_in_expr)
    # print('All done.')


if __name__ =='__main__':

    time_type = "process" #process or total
    reg_names = ["tf-necd", "tf-cod", "sklearn-svddc"] 
    #            "tf-necd", "tf-cod", "pytorch-qrcp", "pytorch-qr", "pytorch-svd", "pytorch-svddc", "sklearn-svddc", "mxnet-svddc",
    data_rows = 1000
            #      '  
    data_cols = 10

    main(time_type, reg_names, data_rows=data_rows, data_cols=data_cols, granularity=2, repeat=3)