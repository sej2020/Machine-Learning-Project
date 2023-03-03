from sklearn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import torch
import mxnet as mx

def read_data(data_path):
    df = pd.read_csv(data_path)
    arr = df.values
    return arr[:, :-1], arr[:, -1]

def gen_cv_samples(X_train, y_train, n_cv_folds):
    """
    Generates a nested array of length k (where k is the number of cv folds)
    Each sub-tuple contains k folds formed into training data and the k+1 fold left out as test data
    
    Args: 
        X_train (nd.array) - Training data already processed
        y_train (nd.array) - Training labels already processed
        
    Returns: 
        train/test data (tuples) - nested_samples gets broken down into four list
    """
    kf = model_selection.KFold(n_splits = n_cv_folds, shuffle = True)
    kf_indices = [(train, test) for train, test in kf.split(X_train, y_train)]
    nested_samples = [(X_train[train_idxs], y_train[train_idxs], X_train[test_idxs], y_train[test_idxs]) for train_idxs, test_idxs in kf_indices]
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for sample in nested_samples:
        for i, var in enumerate((X_tr, y_tr, X_te, y_te)):
            var.append(sample[i])

    return nested_samples
    
def run_linreg(cv_data, regr_name, regr, hyperparams):
    accumulator = []
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(cv_data):
        pred = None

        if regr_name == "sklearn-lst_sq":
            model = regr().fit(X_tr, y_tr, **hyperparams)
            pred = model.predict(X_te)
            
        if regr_name == "sklearn-sgd":
            model = regr(eta0=10**-10, max_iter=10**3, **hyperparams).fit(X_tr, y_tr)
            pred = model.predict(X_te)

        elif regr_name == "tf-lst_sq_fast":
            model = regr(X_tr, y_tr[...,np.newaxis], fast=True, **hyperparams)
            pred = X_te @ model
            
        elif regr_name == "tf-lst_sq_slow":
            model = regr(X_tr, y_tr[...,np.newaxis], fast=False, **hyperparams)
            pred = X_te @ model

        elif regr_name == "pytorch-lst_sq":
            model = regr(torch.Tensor(X_tr), torch.Tensor(y_tr[...,np.newaxis]), **hyperparams).solution
            pred = X_te @ np.array(model)
  
        elif regr_name == "mxnet_lst_sq":
            model = regr(X_tr, y_tr[...,np.newaxis], rcond=None, **hyperparams)[0]
            pred = X_te @ model
                
        
        results = metrics.mean_squared_error(y_te, pred)**(.5)
        accumulator.append(results)
            
    return accumulator
    
def main(data_path, k_folds, data_name):
    X, y = read_data(data_path)
    cv_data = gen_cv_samples(X, y, k_folds)

    regrs = {
            "sklearn-lst_sq": linear_model.LinearRegression,
            "sklearn-sgd": linear_model.SGDRegressor,
            "tf-lst_sq_fast": tf.linalg.lstsq,
            "tf-lst_sq_slow": tf.linalg.lstsq,
            "pytorch-lst_sq":  torch.linalg.lstsq,
            "mxnet_lst_sq": mx.np.linalg.lstsq
    }
    
    result_accumulator = {}
    for name, regr in regrs.items():
        result_accumulator[name] = run_linreg(cv_data, name, regr, {})
    
    results_df = pd.DataFrame(result_accumulator)
    results_df.to_csv("linreg_comparison_output.csv")

    fig, ax = plt.subplots()
    for color, column in zip(['red', 'green', 'blue', 'orange', 'purple', 'yellow'],results_df.columns):
        y = results_df[column]
        x = results_df.index.values
        ax.scatter(x, y, c=color, alpha=0.7, label=column, edgecolors='none', marker='x')

    ax.set_ylabel('RMSE')
    ax.set_xlabel('CV Run')
    ax.legend(loc=(1, 0))
    ax.set_title(f'{data_name} Data')
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(f'MetaAnalysis/testing_linreg_libs/figures/lin_test_{data_name}.png', bbox_inches='tight')

if __name__ == "__main__":
    paths = ["MetaAnalysis/testing_linreg_libs/data/Conductivity.csv", "MetaAnalysis/testing_linreg_libs/data/Concrete.csv", "MetaAnalysis/testing_linreg_libs/data/PowerPlant.csv", "MetaAnalysis/testing_linreg_libs/data/Circle.csv"]
    data_names = ["Conductivity", "Concrete", "PowerPlant", "Circle"]
    for path, data_name in zip(paths, data_names):
        main(data_path = path, k_folds = 10, data_name = data_name)