import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.utils import all_estimators
from time import perf_counter
import multiprocessing as multiprocessing


import warnings
warnings.filterwarnings('ignore')
from inspect import signature, _empty

def get_all_regs() -> list:
    """
    This function imports all sklearn regression estimators. The function will filter all out all regressors
    that take additional parameters. It will return a list of all viable regressor classes and a list of the 
    names of all the viable regressor classes. 
    """
    estimators = all_estimators(type_filter='regressor')
    forbidden_estimators = [
        "DummyRegressor", "GaussianProcessRegressor", "KernelRidge", 
        "QuantileRegressor", "SGDRegressor", 
        "MultiOutputRegressor", "RegressorChain",
        "StackingRegressor", "VotingRegressor","CCA", 
        "IsotonicRegression", "MultiTaskElasticNet", 
        "MultiTaskElasticNetCV", "MultiTaskLasso", 
        "MultiTaskLassoCV", "PLSCanonical"
        ]
    all_regs = []
    all_reg_names = []

    for name, RegressorClass in estimators:
        params = [val[1] for val in signature(RegressorClass).parameters.items()]
        all_optional = True
        for param in params:
            if param.default == _empty:
                all_optional = False
        is_cv_variant = name[-2:] == "CV"
        if all_optional and (name not in forbidden_estimators) and not is_cv_variant:
            print('Appending', name)
            reg = RegressorClass()
            all_regs.append(reg)
            all_reg_names.append(name)
        else:
            print(f"Skipping {name}")
    print(f"List of approved regressors (length {len(all_reg_names)}): {all_reg_names}")
    return all_regs, all_reg_names

def load_data(datapath) -> pd.DataFrame:
    """
    This function will take the relative file path of a csv file and return a pandas DataFrame of the csv content.
    """
    csv_path = os.path.abspath(datapath)
    return pd.read_csv(csv_path)

def create_strat_cat(raw_data) -> pd.DataFrame:
    """
    This function will add a categorical column to the dataframe. This column is the categorical representation of the class
    label of each instance. This will enable the data to be split according to the distribution of the class values. The appended
    dataframe will be returned.
    """
    strat_label = raw_data.columns[-1]
    description = raw_data.describe()
    strat_bins = list(description.loc['min':'max',strat_label])
    strat_bins[0], strat_bins[-1] = -np.inf, np.inf
    raw_data[f"{strat_label}_cat"] = pd.cut(raw_data[strat_label],bins=strat_bins,labels=[1,2,3,4])
    data_w_strat_cat = raw_data
    return data_w_strat_cat, strat_label

def data_split(datapath) -> pd.DataFrame:
    """
    This function will take a relative datapath of a dataset in csv format and will split the data into training attributes, 
    training labels, test attributes, and test labels according to the distribution of a categorical class label.
    """
    raw_data = load_data(datapath)
    data_w_strat_cat, strat_label = create_strat_cat(raw_data)
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(data_w_strat_cat,data_w_strat_cat[f"{strat_label}_cat"]):
        train_set = data_w_strat_cat.loc[train_index]
        test_set = data_w_strat_cat.loc[test_index]
    for set_ in(train_set,test_set):
        set_.drop(f"{strat_label}_cat",axis=1,inplace=True)
    train = train_set.copy()
    test = test_set.copy()

    data_label = train.columns[-1]
    train_attrib = train.drop(data_label,axis=1)
    train_labels = train[data_label].copy()
    test_attrib = test.drop(data_label,axis=1)
    test_labels = test[data_label].copy()

    return train_attrib, train_labels, test_attrib, test_labels

def gen_cv_samples(X_train_df, y_train_df):
    """
    Generates a nested array of length k (where k is the number of cv folds)
    Each sub-tuple contains 9 folds formed into training data and a 10th left out as test data
    
    Args: 
        X_train (nd.array) - Training data already processed
        y_train (nd.array) - Training labels already processed
        
    Returns: 
        nested_samples (tuples)
    """
    X_train, y_train = X_train_df.values, y_train_df.values
    kf = KFold(n_splits = 10, shuffle = True, random_state = 2)
    kf_indices = [(train, test) for train, test in kf.split(X_train, y_train)]
    nested_samples = [(X_train[train_idxs], y_train[train_idxs], X_train[test_idxs], y_train[test_idxs]) for train_idxs, test_idxs in kf_indices]
    
    return nested_samples

def comparison(datapath, n_regressors, metric_list, n_vizualized, metric_help, score_method='neg_mean_squared_errror') -> None:
    """
    This function will perform cross-validation training across multiple regressor types for one dataset. 
    The cross-validation scores will be vizualized in a box plot chart, displaying regressor performance across
    specified metrics. These charts will be saved to the user's CWD as a png file. The best performing model 
    trained on each regressor type will be tested on the set of test instances. The performance of those regs 
    on the test instances will be recorded in a table and saved to the user's CPU as a png file.
    """
    regs, reg_names = get_all_regs()
    if n_regressors != 'all':
        regs, reg_names = regs[0:n_regressors], reg_names[0:n_regressors]
    train_attrib, train_labels, test_attrib, test_labels = data_split(datapath)
    cv_samples = gen_cv_samples(train_attrib, train_labels)
    if score_method not in metric_list:
        metric_list = [score_method]+metric_list
    cv_X_train, cv_y_train, cv_X_test, cv_y_test =              \
        [s[0] for s in cv_samples], [s[1] for s in cv_samples], \
        [s[2] for s in cv_samples], [s[3] for s in cv_samples]
    cv_data = [{name: [] for name in metric_list} for _ in range(len(regs))]
    errors = []
    passed_regs = []
    # training each regressor in CV --- serial#######################################
    # start = perf_counter()
    # for i in range(len(regs) * 10):
    #     x = run(regs[i // 10], reg_names[i // 10], metric_list, metric_help, cv_X_train[i % 10], cv_y_train[i % 10], cv_X_test[i % 10], cv_y_test[i % 10])
    #     if type(x) == list:
    #         for i, (metric, data) in enumerate(cv_data[i // 10].items()):
    #             data.append(x[i])
    #         print(cv_data[i // 10])
    #     else:
    #         errors += [reg_names[i // 10]]
    # print(f"These regressors threw errors in CV: {set(errors)}")
    # stop = perf_counter()
    # print(f"Time to execute regression: {stop - start:.2f}s")
    # return
    # removing the names of the regressors that threw errors in CV###################
    
    # training each regressor in CV --- parallel#####################################
    start = perf_counter()
# # # # # # why integer division for the regressors but modulus for the cv data
    args_lst = [(regs[i // 10], reg_names[i // 10], metric_list, metric_help, cv_X_train[i % 10], cv_y_train[i % 10], cv_X_test[i % 10], cv_y_test[i % 10]) for i in range(len(regs) * 10)]
    multiprocessing.set_start_method("fork", force = True)
    with multiprocessing.Pool() as pool:
        results = pool.starmap(run, args_lst)
              
    for obj in enumerate(results):
        if obj[1] == None:
            errors += [regs[i // 10]]
        else:
            i, (datum, name) = obj
            for i, (metric, data) in enumerate(cv_data[reg_names.index(name)].items()):
                data.append(datum[i])
                
    print(f"These regressors threw errors in CV: {errors}")
    stop = perf_counter()
    print(f"Time to execute regression: {stop - start:.2f}s")
    
    # print(f"parallel: {cv_data}")
    print(f"parallel: {errors}")
    return
    # removing the names of the regressors that threw errors in CV###################

    for j in range(len(regs)):
        if regs[j] not in errors:
            passed_regs += [reg_names[j]]
    figs = [test_best(cv_data, passed_regs, metric_list, test_attrib, test_labels, metric_help, score_method)]
    for metric in metric_list:
        figs += [boxplot(cv_data, passed_regs, metric, n_vizualized, metric_help)]
    for k in range(len(figs)):
        figs[k].savefig(f'fig_{k}.png',bbox_inches='tight')
    pass

def run2(model, metric_list, train_attrib, train_labels) -> dict:
    """
    This function will perform cross-validation training on a given dataset and given regressor. It will return
    a dictionary containing cross-validation performance on various metrics.
    """
    print(f"Checking {model}")
    try:
        cv_outer = KFold(n_splits=6, shuffle=True, random_state=2)
        print([(k, v) for k, v in cv_outer.split(train_attrib)])
        cv_output_dict = cross_validate(model, train_attrib, train_labels, scoring=metric_list, cv=cv_outer, return_estimator=True, n_jobs=3)
        return cv_output_dict
    except:
        pass
    
def run(model, model_name, metric_list, metric_help, train_attrib, train_labels, test_attrib, test_labels) -> dict:
    """
    This function will perform cross-validation training on a given dataset and given regressor. It will return
    a dictionary containing cross-validation performance on various metrics.
    """
    print(f"Checking {model}")
    try:
        model_trained = model.fit(train_attrib, train_labels)
        y_pred = model_trained.predict(test_attrib)
        
        # reg_dict = {model_name:[]}
        # for k in metric_list:
        #     if k == 'neg_root_mean_squared_error':
        #         reg_dict[model_name] += [np.sqrt(metric_help[k][2](test_labels, y_pred))]
        #     else:
        #         reg_dict[model_name] += [metric_help[k][2](test_labels, y_pred)]
        
        reg_dict = []
        for k in metric_list:
            calculated = metric_help[k][2](test_labels, y_pred)
            
            if k == 'neg_root_mean_squared_error':
                reg_dict.append([calculated**.5])
                
            else: reg_dict.append([calculated])
        
        return reg_dict, model_name

    except Exception as e:
        print(e)


def boxplot(cv_data, passed_regs, metric, n_vizualized, metric_help):
    """
    This function will return a box plot chart displaying the cross-validation scores of various regressors for a given metric.
    The box plot chart will be in descending order by median performance. The chart will be saved to the user's CPU as a png file.
    """
    boxfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    #Making CV scores on the specified metric positive and storing in a dataframe. Repeat for each regressor.
    for i,j in zip(cv_data,passed_regs):
            df[j] = list(i['test_'+metric]*metric_help[metric][1])
    #Sorting the columns by median value of the CV scores. The metric_help dictionary helps to determine whether it will be an ascending
    # sort or a descending sort based on the metric.
    sorted_index = df.median().sort_values(ascending=metric_help[metric][0]).index
    df_sorted = df_sorted=df[sorted_index]
    #Creating box plot figure of best n regressors.
    df_sorted.iloc[:,len(df_sorted.columns)-n_vizualized:].boxplot(vert=False,grid=False)
    plt.xlabel(f'CV {metric}')
    plt.ylabel('Models')
    return boxfig

def test_best(cv_data, passed_regs, metric_list, test_attrib, test_labels, metric_help, score_method):
    """
    This function will take the best performing model on each regressor type generated by cross-validation training and 
    apply it to the set of test data. The performance of the regs on the test instances will be displayed on a table and
    saved to the user's CPU as a png file. The regs will be sorted in descending order by performance on specified metrics.
    """
    #initializing a nested list to store the scores of each model on each metric when applied to the test set
    metric_columns = []
    for metric in metric_list:
        metric_columns += [[metric,[]]]
    for i in cv_data:
        #'x' will store the list of CV scores for the given score_method metric
        print([key for key in cv_data.keys()])
        if score_method == 'neg_root_mean_squared_error':
            # x = list(np.sqrt(i['test_'+score_method]*metric_help[score_method][1]))
            x = list(np.sqrt(i[score_method]*metric_help[score_method][1]))

        else:
            # x = list(i['test_'+score_method]*metric_help[score_method][1])
            x = list(i[score_method]*metric_help[score_method][1])
        y = list(i['estimator'])
        #for each score in 'x', if it is the best score, that model will be stored in the 'best' variable
        for j in range(len(x)):
            if metric_help[score_method][0] == True:
                if x[j] == max(x):
                    best = y[j]
            else:
                if x[j] == min(x):
                    best = y[j]
        #the best model will predict the test attributes
        predictions = best.predict(test_attrib)
        #the performance of the model prediction will be stored in the 'metric_columns' list
        # 'metric_help[k[0]][2]' is the associated statistic that will measure the difference between the test labels
        #  and the prediction of the best model
        for k in metric_columns:
            if k[0] == 'neg_root_mean_squared_error':
                k[1] += [round(np.sqrt(metric_help[k[0]][2](test_labels,predictions)),4)]
            else:
                # print(f"{k = }")
                k[1] += [round(metric_help[k[0]][2](test_labels,predictions),4)]
    #preparing dataframe. column names will be the metrics used. the row labels will be the regressors
    columnnames = metric_list
    final_columns = []
    for m in metric_columns:
        final_columns += [m[1]]
    df = pd.DataFrame(np.array(final_columns).T,index=passed_regs,columns=columnnames)
    sorted_df = df.sort_values(by=metric_list[0],ascending=metric_help[metric][0])
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=sorted_df.values, rowLabels=sorted_df.index, colLabels=sorted_df.columns, loc='center')
    fig.tight_layout()
    return fig