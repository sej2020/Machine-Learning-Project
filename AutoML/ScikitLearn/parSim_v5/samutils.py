import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.utils import all_estimators
from sklearn import metrics
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
        train/test data (tuples) - nested_samples gets broken down into four list
    """
    X_train, y_train = X_train_df.values, y_train_df.values
    kf = KFold(n_splits = 10, shuffle = True, random_state = 2)
    kf_indices = [(train, test) for train, test in kf.split(X_train, y_train)]
    nested_samples = [(X_train[train_idxs], y_train[train_idxs], X_train[test_idxs], y_train[test_idxs]) for train_idxs, test_idxs in kf_indices]
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for sample in nested_samples:
        for i, var in enumerate((X_tr, y_tr, X_te, y_te)):
            var.append(sample[i])
    
    return (X_tr, y_tr, X_te, y_te)

def comparison(datapath, n_regressors, metric_list, n_vizualized, metric_help, score_method='Root Mean Squared Error') -> None:
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

    metric_list = [score_method] + metric_list
    for i, item in enumerate(metric_list[1:]):
        if item == metric_list[0]:
            del metric_list[i+1]

    cv_X_train, cv_y_train, cv_X_test, cv_y_test = gen_cv_samples(train_attrib, train_labels)
    start = perf_counter()
    args_lst = [(regs[i // 10], reg_names[i // 10], metric_list, metric_help, cv_X_train[i % 10], cv_y_train[i % 10], cv_X_test[i % 10], cv_y_test[i % 10]) for i in range(len(regs) * 10)]
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool() as pool:
        results = pool.starmap(run, args_lst)
                         
    org_results = {} # -> {'Reg Name': [{'Same Reg Name': [metric, metric, ..., Reg Obj.]}, {}, {}, ... ], '':[], '':[], ... } of raw results
    for r in results:
        if type(r) == dict:
            if list(r.keys())[0] in org_results:
                org_results[list(r.keys())[0]] += [r]
            else:
                org_results[list(r.keys())[0]] = [r]

    fin_org_results = {} # -> {'Reg Name': [{'Same Reg Name': [metric, metric, ..., Reg Obj.]}, {}, {}, ... ], '':[], '':[], ... } of only successful CV runs
    for k,v in org_results.items():
        if len(v) == 10:
            fin_org_results[k] = v

    stop = perf_counter()
    print(f"Time to execute regression: {stop - start:.2f}s")

    # figs = [test_best(results, metric_list, test_attrib, test_labels, metric_help)]
    figs = [test_best(fin_org_results, metric_list, test_attrib, test_labels, metric_help)]

    # for index in len(range(metric_list)):
    #     figs += [boxplot(results, reg_names, metric_list, n_vizualized, metric_help, index)]

    for k in range(len(figs)):
        figs[k].savefig(f'AutoML/ScikitLearn/parSim_v5/par_1/figure_{k}.png', bbox_inches='tight', dpi=600.0)

    return f'time: {stop-start}, fin_org_results length: {len(fin_org_results)}'
    

def run(model, model_name, metric_list, metric_help, train_attrib, train_labels, test_attrib, test_labels) -> dict:
    """
    This function will perform cross-validation training on a given dataset and given regressor. It will return
    a dictionary containing cross-validation performance on various metrics.
    """
    print(f"Checking {model}")
    try:
        model_trained = model.fit(train_attrib, train_labels)
        y_pred = model_trained.predict(test_attrib)
        reg_dict = {model_name: []}
        for k in metric_list:
            calculated = metric_help[k][2](test_labels, y_pred)
            reg_dict[model_name].append(calculated if k != 'Root Mean Squared Error' else calculated**.5)
        reg_dict[model_name].append(model_trained)
        return reg_dict

    except Exception as e:
        print(e)
        pass


def boxplot(results, reg_names, metric_list, n_vizualized, metric_help, index):
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


def test_best(fin_org_results, metric_list, test_attrib, test_labels, metric_help):
    """
    This function will take the best performing model on each regressor type generated by cross-validation training and 
    apply it to the set of test data. The performance of the regs on the test instances will be displayed on a table and
    saved to the user's CPU as a png file. The regs will be sorted in descending order by performance on specified metrics.
    """
    columns = metric_list
    rows = []
    output = []
    for k,v in fin_org_results.items():
        rows.append(k)
        scores = []
        models = []
        for dict in v:
            scores.append(list(dict.values())[0][0])
            models.append(list(dict.values())[0][-1])
        if metric_help[metric_list[0]][0] == True:
            best = max(zip(scores, models), key = lambda pair: pair[0])[1]
        else:
            best = min(zip(scores, models), key = lambda pair: pair[0])[1]

        best_predict = best.predict(test_attrib)

        single_reg_output = []
        for m in metric_list:
            calculated = metric_help[m][2](test_labels, best_predict)
            single_reg_output.append(round(calculated if m != 'Root Mean Squared Error' else calculated**.5,4))
        
        output.append(single_reg_output)

    df = pd.DataFrame(data=output, index=rows, columns=columns)

    sorted_df = df.sort_values(by=columns[0], axis=0, ascending=not(metric_help[columns[0]][0]))
    print(sorted_df)

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=sorted_df.values, rowLabels=sorted_df.index, colLabels=sorted_df.columns, loc='center')
    fig.tight_layout()
    return fig