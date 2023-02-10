from sklearn import *
import numpy as np
import pandas as pd
import tensorflow as tf
# import torch
# import mxnet as mx

def read_data(data_path):
    df = pd.read_csv(data_path)
    arr = df.values
    return arr[:, :-1], arr[:, -1]

def main(data_path):
    X, y = read_data(data_path)
    x_tr, x_te, y_tr, y_te = model_selection.train_test_split(X,y, test_size=0.2)
    model = linear_model.SGDRegressor().fit(x_tr, y_tr)
    modellin = linear_model.LinearRegression().fit(x_tr, y_tr)
    pred = model.predict(x_te)
    predlin = modellin.predict(x_te)
    results = metrics.mean_squared_error(y_te, pred)**(.5)
    resultslin = metrics.mean_squared_error(y_te, predlin)**(.5)
    return results, resultslin

    
if __name__ == "__main__":
    print(main(data_path = "AutoML\PowerPlantData\Folds5x2_pp.csv"))
