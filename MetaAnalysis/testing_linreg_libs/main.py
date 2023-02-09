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
            if regr == "least_squares":
                model = regr(X_tr, y_tr, **hyperparams)
                pred = X_te @ model
            elif regr == "stochastic_gradient_descent":
                pass

        elif regr_lib == "pytorch":
            if regr == "least_squares":
                model = regr(X_tr, y_tr, **hyperparams)
                pred = X_te @ model
            elif regr == "stochastic_gradient_descent":
                pass
                
        elif regr_lib == "mxnet":
            if regr == "least_squares":
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
            "stochastic_gradient_descent": tf.keras.optimizers.experimental.SGD
        },
        "pytorch": {
            "least_squares":  torch.linalg.lstsq,
            "stochastic_gradient_descent": torch.optim.SGD
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



#Do we need to do SGD for Torch and TF? It will be a lot more code. Here's something I ripped from the internet:
#------------------------------------------------------------------------------------------------------------------


# Creating a function f(X) with a slope of -5
# X = torch.arange(-5, 5, 0.1).view(-1, 1)
# func = -5 * X

# # Adding Gaussian noise to the function f(X) and saving it in Y
# Y = func + 0.4 * torch.randn(X.size())


# # defining the function for forward pass for prediction
# def forward(x):
#     return w * x + b

# # evaluating data points with Mean Square Error (MSE)
# def criterion(y_pred, y):
#     return torch.mean((y_pred - y) ** 2)

# # Batch gradient descent
# w = torch.tensor(-10.0, requires_grad=True)
# b = torch.tensor(-20.0, requires_grad=True)
# step_size = 0.1
# loss_BGD = []
# n_iter = 20

# for i in range (n_iter):
#     # making predictions with forward pass
#     Y_pred = forward(X)
#     # calculating the loss between original and predicted data points
#     loss = criterion(Y_pred, Y)
#     # storing the calculated loss in a list
#     loss_BGD.append(loss.item())
#     # backward pass for computing the gradients of the loss w.r.t to learnable parameters
#     loss.backward()
#     # updateing the parameters after each iteration
#     w.data = w.data - step_size * w.grad.data
#     b.data = b.data - step_size * b.grad.data
#     # zeroing gradients after each iteration
#     w.grad.data.zero_()
#     b.grad.data.zero_()
#     # priting the values for understanding
#     print('{}, \t{}, \t{}, \t{}'.format(i, loss.item(), w.item(), b.item()))

# # Stochastic gradient descent
# w = torch.tensor(-10.0, requires_grad=True)
# b = torch.tensor(-20.0, requires_grad=True)
# step_size = 0.1
# loss_SGD = []
# n_iter = 20

# for i in range(n_iter):  
#     # calculating true loss and storing it
#     Y_pred = forward(X)
#     # store the loss in the list
#     loss_SGD.append(criterion(Y_pred, Y).tolist())

#     for x, y in zip(X, Y):
#     # making a pridiction in forward pass
#     y_hat = forward(x)
#     # calculating the loss between original and predicted data points
#     loss = criterion(y_hat, y)
#     # backward pass for computing the gradients of the loss w.r.t to learnable parameters
#     loss.backward()
#     # updateing the parameters after each iteration
#     w.data = w.data - step_size * w.grad.data
#     b.data = b.data - step_size * b.grad.data
#     # zeroing gradients after each iteration
#     w.grad.data.zero_()
#     b.grad.data.zero_()
