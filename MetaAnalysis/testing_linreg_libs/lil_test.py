from sklearn import *
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
# import torch
# import mxnet as mx

def read_data(data_path):
    df = pd.read_csv(data_path)
    arr = df.values
    return arr[:, :-1], arr[:, -1]

def main(data_path):
    X, y = read_data(data_path)
    x_tr, x_te, y_tr, y_te = model_selection.train_test_split(X,y, test_size=0.2)
    n_trials = 5
    etapows = range(0, -15, -1)
    data = [[] for _ in etapows]
    for i, pow in enumerate(etapows):
        print(f"EXP: {pow}")
        for _ in range(n_trials):
            eta = 10**pow
            model = linear_model.SGDRegressor(eta0=eta).fit(x_tr, y_tr)
            pred = model.predict(x_te)
            results = metrics.mean_squared_error(y_te, pred)**(.5)
            data[i] += [results]
                
    modellin = linear_model.LinearRegression().fit(x_tr, y_tr)
    predlin = modellin.predict(x_te)
    resultslin = metrics.mean_squared_error(y_te, predlin)**(.5)
    arr_avg_sgds = np.array(data).mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(etapows, arr_avg_sgds)
    ax.set_xticks(etapows)
    ax.set_xticklabels([str(x) for x in etapows])
    ax.set_xlabel("Eta Exponent (10^eta_exp)")
    ax.set_ylabel("RMSE")
    ax.hlines(resultslin, -15, 0)
    plt.yscale("log")
    plt.show()
    
    return results, resultslin

    
if __name__ == "__main__":
    sgd, lstq = main(data_path = "AutoML/ConcreteData/Concrete_Data.csv")
    print(f"SGD RMSE: {sgd}\nLeast Squares RMSE: {lstq}\nRatio: {sgd/lstq}")
