from sklearn import *
import numpy as np
import pandas as pd
import tensorflow as tf
import pytorch as torch
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
    
def run_linreg(cv_data, regr, regr_lib, hyperparams):
    accumulator = []
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(cv_data):

        if regr_lib == "sklearn":
            model = regr().fit(X_tr, y_tr, **hyperparams)
            pred = model.predict(X_te)

        elif regr_lib == "tensorflow":
            model = regr(X_tr, y_tr, **hyperparams)
            pred = X_te @ model

        elif regr_lib == "pytorch":
            model = regr(X_tr, y_tr, **hyperparams)
            pred = X_te @ model
  
        elif regr_lib == "mxnet":
            model = regr(X_tr, y_tr, **hyperparams)
            pred = X_te @ model
                
            
        results = metrics.mean_squared_error(y_te, pred)**(.5)
        accumulator.append(results)
            
    return accumulator
    
def main(data_path, libs, methods, k_folds):
    X, y = read_data(data_path)
    cv_data = gen_cv_samples(X, y, k_folds)

    regrs = {
        "sklearn": {
            "least_squares": linear_model.LinearRegression,
            "stochastic_gradient_descent": linear_model.SGDRegressor
        },
        "tensorflow": {
            "least_squares": tf.linalg.lstsq,
        },
        "pytorch": {
            "least_squares":  torch.linalg.lstsq,
        },
        "mxnet": {
            "least_squares": mx.np.linalg.lstsq
            }
    }
    
    result_accumulator = {}
    
    for lib in libs:
        for method in methods:
            results = run_linreg(cv_data, regrs[lib][method], lib, {})
            result_accumulator[f"{lib}-{method}"] = results
    
    results_df = pd.DataFrame(result_accumulator)
    print(results_df)

    
if __name__ == "__main__":
    main(
        data_path = "AutoML/ConcreteData/Concrete_Data.csv",
        libs = ("sklearn",),
        methods = ("least_squares",),
        
        k_folds = 10,
    )


