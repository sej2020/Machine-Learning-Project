from sklearn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# import torch
# import mxnet as mx

def read_data(data_path):
    df = pd.read_csv(data_path)
    arr = df.values
    return arr[:, 0].reshape((len(arr),1)), arr[:, 1].reshape((len(arr),1))

def run_linreg(X, y, regr_name, regr):

    if regr_name == "sklearn-lst_sq":
        model = regr().fit(X,y)
        pred = model.predict(X)
        
    elif regr_name == "sklearn-sgd":
        model = regr(eta0=10**-10, max_iter=10**3).fit(X, y)
        pred = model.predict(X)

    elif regr_name == "tf-lst_sq_fast":
        print(X, y, np.ravel(y))
        model = regr(X, y, fast=True)
        pred = X @ model
        
    elif regr_name == "tf-lst_sq_slow":
        model = regr(X, y, fast=False)
        pred = X @ model

    elif regr_name == "pytorch-lst_sq":
        model = regr(torch.Tensor(X), torch.Tensor(y[...,np.newaxis])).solution
        pred = X @ np.array(model)

    elif regr_name == "mxnet_lst_sq":
        model = regr(X, y[...,np.newaxis], rcond=None)[0]
        pred = X @ model
                
    results = [X, pred]
            
    return results
    

def main(path):
    X, y = read_data(path)

    regrs = {
            "sklearn-lst_sq": linear_model.LinearRegression,
            "sklearn-sgd": linear_model.SGDRegressor,
            "tf-lst_sq_fast": tf.linalg.lstsq,
            "tf-lst_sq_slow": tf.linalg.lstsq
            # "pytorch-lst_sq":  torch.linalg.lstsq,
            # "mxnet_lst_sq": mx.np.linalg.lstsq
    }
    
    result_accumulator = {}
    for name, regr in regrs.items():
        result_accumulator[name] = run_linreg(X, y, name, regr)
    
    fig, ax1 = plt.subplots()

    for name, results in result_accumulator.items():
        ax1.plot(results[0], results[1])

    ax1.scatter(X, y)
    ax1.set_ylabel("y")
    ax1.set_xlabel("x")
    fig.legend([k for k in result_accumulator.keys()])

    fig.set_dpi(100)
    plt.gca().set_aspect('equal')

    plt.show()
    return

if __name__ == "__main__":
    path = "MetaAnalysis/testing_linreg_libs/data/Circle.csv"
    main(path=path)