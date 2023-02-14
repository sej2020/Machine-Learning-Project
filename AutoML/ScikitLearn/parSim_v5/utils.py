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


def get_all_regs(which_regressors: dict) -> list & list:
    """
    This function imports all sklearn regression estimators. The function will filter all out all regressors
    that take additional parameters. It will return a list of all viable regressor classes and a list of the 
    names of all the viable regressor classes. 
    
    Args:
        which_regressors (dict) - dictionary of key:value pairs of form <'RegressorName'> : <Bool(0)|Bool(1)>

    Returns:
        regressors (lists) - two lists, the first being all regressor objects, the seconds being the corresponding regressor names
    """

    #importing all sklearn regressors and establishing which regressors will be ommited from the run
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

    #removing regressors that require additional parameters or those that are cross-validation variants of existing regressors
    for name, RegressorClass in estimators:
        params = [val[1] for val in signature(RegressorClass).parameters.items()]
        all_optional = True
        for param in params:
            if param.default == _empty:
                all_optional = False
        is_cv_variant = name[-2:] == "CV"
        if all_optional and (name not in forbidden_estimators) and not is_cv_variant and which_regressors[name] == 1:
            print('Appending', name)
            reg = RegressorClass()
            all_regs.append(reg)
            all_reg_names.append(name)
        else:
            print(f"Skipping {name}")
    print(f"List of approved regressors (length {len(all_reg_names)}): {all_reg_names}")
    return all_regs, all_reg_names


def load_data(datapath: str) -> pd.DataFrame:
    """
    This function will take the relative file path of a csv file and return a pandas DataFrame of the csv content.
    
    Args:
        datapath (str) - a file path (eventually from s3 bucket) of the csv data
    
    Returns:
        raw data (pd.DataFrame) - a pandas dataframe containing the csv data
    """

    csv_path = os.path.abspath(datapath)
    return pd.read_csv(csv_path)


def create_strat_cat(raw_data: pd.DataFrame) -> pd.DataFrame & str:
    """
    This function will add a categorical column to the dataframe. This column is the categorical representation of the class
    label of each instance. This will enable the data to be split according to the distribution of the class values. The appended
    dataframe will be returned.

    Args:
        raw_data (pd.DataFrame) - a pandas dataframe containing raw data

    Returns:
        data with a stratified label category - the raw data as a pandas dataframe, with a column containing discretized label data
        stratified category name - the name of the final column of this pandas dataframe
    """

    strat_label = raw_data.columns[-1]
    description = raw_data.describe()
    strat_bins = list(description.loc['min':'max',strat_label])
    strat_bins[0], strat_bins[-1] = -np.inf, np.inf
    raw_data[f"{strat_label}_cat"] = pd.cut(raw_data[strat_label],bins=strat_bins,labels=[1,2,3,4])
    data_w_strat_cat = raw_data
    return data_w_strat_cat, strat_label


def data_split(datapath: str, test_set_size: float) -> tuple:
    """
    This function will take a relative datapath of a dataset in csv format and will split the data into training attributes, 
    training labels, test attributes, and test labels according to the distribution of a categorical class label.
    
    Args:
        datapath (str) - a file path (eventually from s3 bucket) of the csv data
        test_set_size (float) - a number between 0 and 1 that indicates the proportion of data to be allocated to the test set (Default: 0.2)
    
    Returns:
        train/test datasets (tuple) - Four pandas dataframes: the first is training set attributes, the second is training set
                                      labels, the third is test set attributes, the fourth is test set labels
    """

    #the data is loaded and the label is discretized in order to create a stratified train-test split
    raw_data = load_data(datapath)
    data_w_strat_cat, strat_label = create_strat_cat(raw_data)

    #the training and test sets are created
    split = StratifiedShuffleSplit(n_splits=1,test_size=test_set_size)
    for train_index, test_index in split.split(data_w_strat_cat,data_w_strat_cat[f"{strat_label}_cat"]):
        train_set = data_w_strat_cat.loc[train_index]
        test_set = data_w_strat_cat.loc[test_index]
    for set_ in(train_set,test_set):
        set_.drop(f"{strat_label}_cat",axis=1,inplace=True)
    train = train_set.copy()
    test = test_set.copy()

    #the training and test sets are further split into attribute and label sets
    data_label = train.columns[-1]
    train_attrib = train.drop(data_label,axis=1)
    train_labels = train[data_label].copy()
    test_attrib = test.drop(data_label,axis=1)
    test_labels = test[data_label].copy()

    return (train_attrib, train_labels, test_attrib, test_labels)


def gen_cv_samples(X_train_df: pd.DataFrame, y_train_df: pd.DataFrame, n_cv_folds: int) -> tuple:
    """
    Generates a nested array of length k (where k is the number of cv folds).
    Each sub-tuple contains k folds formed into training data and the k+1 fold left out as test data.
    
    Args: 
        X_train_df (pd.DataFrame) - training data already processed
        y_train (pd.DataFrame) - training labels already processed
        n_cv_folds (int) - the number of folds for k-fold cross validation training (Default: 10)
        
    Returns: 
        train/test data (tuples) - nested_samples gets broken down into four lists
    """
    
    X_train, y_train = X_train_df.values, y_train_df.values
    kf = KFold(n_splits = n_cv_folds, shuffle = True)
    kf_indices = [(train, test) for train, test in kf.split(X_train, y_train)]
    nested_samples = [(X_train[train_idxs], y_train[train_idxs], X_train[test_idxs], y_train[test_idxs]) for train_idxs, test_idxs in kf_indices]
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for sample in nested_samples:
        for i, var in enumerate((X_tr, y_tr, X_te, y_te)):
            var.append(sample[i])
    return (X_tr, y_tr, X_te, y_te)


def metric_help_func():
    """
    Internal table to assist with any functions involving metrics
    Args: 
        None 
    Returns: 
        metric_table (dict) - dictionary of general form: { 'metric': [ higher score is better?, positive or negative score values, accociated stat function ] } 
    """
    
    metric_table = {'Explained Variance': [True, 1, metrics.explained_variance_score], 'Max Error': [False, 1, metrics.max_error],
                            'Mean Absolute Error': [False, -1, metrics.mean_absolute_error], 'Mean Squared Error': [False, -1, metrics.mean_squared_error],
                            'Root Mean Squared Error': [False, -1, metrics.mean_squared_error], 'Mean Squared Log Error': [False, -1, metrics.mean_squared_log_error],
                            'Median Absolute Error': [False, -1, metrics.median_absolute_error], 'R-Squared': [True, 1, metrics.r2_score],
                            'Mean Poisson Deviance': [False, -1, metrics.mean_poisson_deviance], 'Mean Gamma Deviance': [False, -1, metrics.mean_gamma_deviance],
                            'Mean Absolute Percentage Error': [False, -1, metrics.mean_absolute_percentage_error], 'D-Squared Absolute Error Score': [True, 1, metrics.d2_absolute_error_score],
                            'D-Squared Pinball Score': [True, 1, metrics.d2_pinball_score], 'D-Squared Tweedie Score': [True, 1, metrics.d2_tweedie_score]
                            }
    return metric_table

def comparison(datapath: str, which_regressors: dict, metric_list: list, styledict: dict, n_vizualized_bp=-1, n_vizualized_tb=-1, test_set_size=0.2, n_cv_folds=10, score_method='Root Mean Squared Error') -> None:
    """
    This function will perform cross-validation training across several regressor types for one dataset. 
    The cross-validation scores will be recorded and vizualized in a box plot chart, displaying regressor performance across
    specified metrics. These charts will be saved to the user's CWD as a png file. The best performing model 
    trained on each regressor type will be tested on the set of test instances. The performance of those regs 
    on the test instances will be recorded in a table and saved to the user's CPU as a png file.
    
    Args:
        datapath (str) - a file path (eventually from s3 bucket) of the csv data
        which_regressors (dict) - dictionary of key:value pairs of form <'RegressorName'> : <Bool(0)|Bool(1)>
        metric_list (list) - the regressors will be evaluated on these metrics during cross-validation and visualized
        styledict (dict) - container for user to specify style of boxplots
        n_vizualized_bp (int) - the top scoring 'n' regressors in cross-validation to be included in boxplot visualizations. The value -1 will include all regressors (Default: -1)
        n_vizualized_tb (int) - the top scoring 'n' regressors over the test set to be included in final table. The value -1 will include all regressors (Default: -1)
        test_set_size (float) - a number between 0 and 1 that indicates the proportion of data to be allocated to the test set (Default: 0.2)
        n_cv_folds (int) - the number of folds for k-fold cross validation training (Default: 10)
        score_method (str) - the regressors will be evaluated on this metric to determine which regressors perform best (Default: 'Root Mean Squared Error')
    
    Returns:
        Eventually will put csv results into s3 bucket. may or may not provide visualizations
    """

    regs, reg_names = get_all_regs(which_regressors)
    train_attrib, train_labels, test_attrib, test_labels = data_split(datapath, test_set_size)

    #appending the score method to the metric list to be used in the remainder of the program
    metric_list = [score_method] + metric_list
    for i, item in enumerate(metric_list[1:]):
        if item == metric_list[0]:
            del metric_list[i+1]

    metric_help = metric_help_func()

    #creating cv samples and running each regressor over these samples
    cv_X_train, cv_y_train, cv_X_test, cv_y_test = gen_cv_samples(train_attrib, train_labels, n_cv_folds)
    start = perf_counter()
    args_lst = [(regs[i // n_cv_folds], reg_names[i // n_cv_folds], metric_list, metric_help, cv_X_train[i % n_cv_folds], cv_y_train[i % n_cv_folds], cv_X_test[i % n_cv_folds], cv_y_test[i % n_cv_folds]) for i in range(len(regs) * n_cv_folds)]
    multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.starmap(run, args_lst)

    #organizing results of cv runs into a dictionary                   
    org_results = {} # -> {'Reg Name': [{'Same Reg Name': [metric, metric, ..., Reg Obj.]}, {}, {}, ... ], '':[], '':[], ... } of raw results
    for single_reg_dict in results:
        if type(single_reg_dict) == dict:
            reg_name = list(single_reg_dict.keys())[0]
            if reg_name in org_results:
                org_results[reg_name] += [single_reg_dict]
            else:
                org_results[reg_name] = [single_reg_dict]

    #keeping only those results that did not throw an error during any cv run
    fin_org_results = {k: v for k,v in org_results.items() if len(v) == n_cv_folds}

    stop = perf_counter()
    print(f"Time to execute regression: {stop - start:.2f}s")

    #generating figures and saving to the user's CWD
    figs = [test_best(fin_org_results, metric_list, test_attrib, test_labels, metric_help, n_vizualized_tb)]
    for index in range(len(metric_list)):
        figs += [boxplot(fin_org_results, styledict, metric_list, metric_help, n_vizualized_bp, index)]
    for k in range(len(figs)):
        figs[k].savefig(f'AutoML/ScikitLearn/parSim_v5/par_1/figure_{k}.png', bbox_inches='tight', dpi=styledict['dpi'])
    pass
    

def run(reg: object, reg_name: str, metric_list, metric_help, train_attrib, train_labels, test_attrib, test_labels) -> dict:
    """
    This function will perform cross-validation training on a given dataset and given regressor. It will return
    a dictionary containing cross-validation performance on various metrics.
    
    Args:
        reg (object) - a scikit-learn regressor object
        reg_name (str) - the associated scikit-learn regressor name
        metric_list (list) - the regressors will be evaluated on these metrics during cross-validation and visualized
        metric_help (dict) - a dictionary to assist with any functions involving metrics
        train_attrib (list) - list of training attributes
        train_labels (list)
        test_attrib (list)
        test_labels (list)

    Returns:
        Eventually will put csv results into s3 bucket. may or may not provide visualizations

    """
    print(f"Checking {reg}")
    try:
        model_trained = reg.fit(train_attrib, train_labels)
        y_pred = model_trained.predict(test_attrib)
        reg_dict = {reg_name: []}
        for k in metric_list:
            calculated = metric_help[k][2](test_labels, y_pred)
            reg_dict[reg_name].append(calculated if k != 'Root Mean Squared Error' else calculated**.5)
        reg_dict[reg_name].append(model_trained)
        return reg_dict

    except Exception as e:
        print(e)
        pass


def boxplot(fin_org_results, styledict, metric_list, metric_help, n_vizualized_bp, index):
    """
    This function will return a box plot chart displaying the cross-validation scores of various regressors for a given metric.
    The box plot chart will be in descending order by median performance. The chart will be saved to the user's CPU as a png file.
    """
    boxfig = plt.figure(constrained_layout=True)

    metric = metric_list[index]
    df = pd.DataFrame()
    for k,v in fin_org_results.items():
        df[k] = [list(dict.values())[0][index] for dict in v]

    #Sorting the columns by median value of the CV scores. The metric_help dictionary helps to determine whether it will be an ascending
    # sort or a descending sort based on the metric.
    sorted_index = df.median().sort_values(ascending=metric_help[metric][0]).index
    df_sorted = df[sorted_index]

    #Creating box plot figure of best n regressors.
    df_final = df_sorted.iloc[:,len(df_sorted.columns)-n_vizualized_bp:]
    bp_data = []
    for column in df_final.columns:
        bp_data.append(df[column[:]].tolist())

    boxfig = plt.figure()
    ax = boxfig.add_subplot(111)
    bp = ax.boxplot(bp_data, patch_artist = True, vert = 0, boxprops = styledict['boxprops'],
                    flierprops = styledict['flierprops'], medianprops = styledict['medianprops'],
                    whiskerprops = styledict['whiskerprops'], capprops = styledict['capprops']
                    )
    
    for patch in bp['boxes']:
        patch.set_facecolor(styledict['boxfill'])
   
    ax.set_yticklabels([column for column in df_final.columns])
    ax.yaxis.grid(styledict['grid'])
    ax.xaxis.grid(styledict['grid'])
    
    plt.title("Cross Validation Scores")

    ax.set_xlabel(f'{metric}')
    ax.set_ylabel('Models')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    return boxfig


def test_best(fin_org_results, metric_list, test_attrib, test_labels, metric_help, n_vizualized_tb):
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
        
        scores = [list(dict.values())[0][0] for dict in v]
        models = [list(dict.values())[0][-1] for dict in v]

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

    df_sorted = df.sort_values(by=columns[0], axis=0, ascending=not(metric_help[columns[0]][0]))
    print(df_sorted)

    df_sorted = df_sorted.iloc[:n_vizualized_tb]

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=df_sorted.values, rowLabels=df_sorted.index, colLabels=df_sorted.columns, loc='center')
    fig.tight_layout()
    return fig