from sklearn import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import mxnet as mx
import sys
from time import perf_counter, process_time

def linreg_pipeline(data, include_regs="all", split_pcnt=None, random_seed=None, time_type="total", generate_figures="all", vis_theme="darkgrid"):
    """
    Pipeline takes actual data, not a path, and runs each of the linear regression algorithms available over it
    
    Args: 
        data - pd.DataFrame or np.ndarray with target variable in final column and no categorical or missing data
        
        include_regs - "all" to use all algorithms or a list of desired algorithms to use a subset
            options: "sklearn-lst_sq" ::: "tf-lst_sq_fast" ::: "tf-lst_sq_slow" ::: "pytorch-lst_sq" ::: "mxnet_lst_sq"
            
        split_pcnt - None to train and test the algorithm over the entirety of the data or a real number from 1 - 100 to use that percentage of the data as a training set and test on the remainder
        
        random_seed - only used by sklearn.model_selection.train_test_split if split_pcnt != None
        
        time_type - "total" to use perf_counter and measure time in sleep, or "process" to measure only cpu time with process_time
        
        generate_figures - "all" to build all figure types below, False to skip generating any figures, or a list of desired figures to use a subset
            options: "regressor_barchart" ::: "scatterplot_w_regression_line"
            
        vis_theme - "darkgrid" by default, or specify any one of the below options
            optiosn: "darkgrid" ::: "whitegrid" ::: "dark" ::: "white" ::: "ticks"
        
    Returns result_dict:
        includes various useful things
        
    """
    ### Validate data ###
    assert isinstance(data, (pd.DataFrame, np.ndarray)), f"Data must be of type pd.DataFrame or np.ndarray, not {type(data)}"
    if isinstance(data, pd.DataFrame):
        data = data.values
    assert data.dtype != "object", "Data must be numeric, not object type - remove categorical data, impute missing values, or make array regular (not ragged)"
    #####################
    
    ### Set time type ###
    match time_type:
        case "total":
            timer = perf_counter
            
        case "process":
            timer = process_time
    #####################
    
    ### Build list of regressors to run ###
    possible_regressors = [
        "sklearn-lst_sq",
        "tf-lst_sq_fast",
        "tf-lst_sq_slow",
        "pytorch-lst_sq",
        "mxnet_lst_sq",
    ]
    if include_regs == "all":
        reg_names = possible_regressors
        
    elif isinstance(include_regs, (tuple, list, set)):
        reg_names = list({reg_name for reg_name in include_regs if reg_name in possible_regressors})
        
    else: 
        raise ValueError(f'Invalid value passed for include_regs: {include_regs}\nSee documentation')
    #######################################
    
    ### Split data if needed ###
    assert isinstance(split_pcnt, (int, float)) or split_pcnt is None, f"Invalid value passed for split_pcnt: {split_pcnt}\nSee documentation"
    if split_pcnt is None:
        tt_split_flag = False # if no split_pcnt is passed, should train and test on same data - also affects visualization
        X_train, X_test, y_train, y_test = data[:, :-1], data[:, :-1], data[:, -1], data[:, -1]
        
    else:
        tt_split_flag = True 
        X_train, X_test, y_train, y_test = model_selection.train_test_split(data, train_size = (split_pcnt / 100), random_state=random_seed)
    ############################
    
    ### Determine which figures will be made ###
    possible_figures = [
        "regressor_barchart",
        "scatterplot_w_regression_line",
    ]
    if generate_figures == "all":
        figures = possible_figures
        
    elif isinstance(generate_figures, (tuple, list, set)):
        figures = [figure_name for figure_name in generate_figures if figure_name in possible_figures]
        
    elif generate_figures == False:
        figures = None
        
    else: 
        raise ValueError(f'Invalid value passed for generate_figures: {generate_figures}\nSee documentation')
    ############################################

    ### Actual linear regression loop ###
    results_dict = {}
        
    for reg_name in reg_names:       

        start_lstsq = timer()
        
        if reg_name == "sklearn-lst_sq":
            model = linear_model.LinearRegression().fit(X_train,y_train)
            pred = model.predict(X_test)

        elif reg_name == "tf-lst_sq_fast":
            model = tf.linalg.lstsq(X_train, y_train[...,np.newaxis], fast=True)
            pred = X_test @ model
            
        elif reg_name == "tf-lst_sq_slow":
            model = tf.linalg.lstsq(X_train, y_train[...,np.newaxis], fast=False)
            pred = X_test @ model

        elif reg_name == "pytorch-lst_sq":
            model = torch.linalg.lstsq(torch.Tensor(X_train), torch.Tensor(y_train[...,np.newaxis])).solution
            pred = X_test @ np.array(model)

        elif reg_name == "mxnet_lst_sq":
            model = mx.np.linalg.lstsq(X_train, y_train[...,np.newaxis], rcond=None)[0]
            pred = X_test @ model
            
        stop_lstsq = timer()

        results_dict[reg_name] = {
            "model": model,
            "y_pred": pred,
            "elapsed_time": stop_lstsq - start_lstsq
            }
    ########################################################################
    
    ### Process results from run ###
    successful_regs = list(results_dict.keys())
    for reg_name, reg_output in results_dict.items():
        
    ################################
    
    ### Plotting desired figures ###
    possible_themes = ["darkgrid", "whitegrid", "dark", "white", "ticks"]
    assert vis_theme in possible_themes, f"Invalid value passed for vis_theme: {vis_theme}\nSee documentation"
    sns.set_theme(vis_theme)
    
    if "regressor_barchart" in figures:
        fig, ax = plt.subplots()
        sns.barplot(x=)
    ################################
    
    return results_dict
    

def main(path, params):
    data = pd.read_csv(path).values
    results = linreg_pipeline(data, **params)
    print(results)

if __name__ == "__main__":
    main(
        path = "BetaDataExper/Circle/Circle.csv",
        params = {
            "random_seed": 100,
        }
    )
