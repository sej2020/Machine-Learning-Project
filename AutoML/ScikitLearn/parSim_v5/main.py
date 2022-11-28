import multiprocessing as mp
import os
import sys
sys.path.append(os.getcwd())
from utils import *


def comparison(datapath, n_regressors, metric_list, n_vizualized, metric_help, score_method='neg_mean_squared_errror') -> None:
    """
    This function will perform cross-validation training across multiple regressor types for one dataset. 
    The cross-validation scores will be vizualized in a box plot chart, displaying regressor performance across
    specified metrics. These charts will be saved to the user's CPU as a png file. The best performing model 
    trained on each regressor type will be tested on the set of test instances. The performance of those regs 
    on the test instances will be recorded in a table and saved to the user's CPU as a png file.
    """
    regs, reg_names = get_all_regs()
    if n_regressors != 'all':
        regs, reg_names = regs[0:n_regressors], reg_names[0:n_regressors]
    train_attrib, train_labels, test_attrib, test_labels = data_split(datapath)
    cv_data = []
    errors = []
    passed_regs = []
    if score_method not in metric_list:
        metric_list = [score_method]+metric_list
    #training each regressor in CV
    for i in range(len(regs)):
        x = run(regs[i], metric_list, train_attrib, train_labels)
        if type(x) == dict:
            cv_data += [x]
        else:
            errors += [regs[i]]
    print(f"These regressors threw errors in CV: {errors}")
    #removing the names of the regressors that threw errors in CV
    print(cv_data)
    for j in range(len(regs)):
        if regs[j] not in errors:
            passed_regs += [reg_names[j]]
    figs = [test_best(cv_data, passed_regs, metric_list, test_attrib, test_labels, metric_help, score_method)]
    for metric in metric_list:
        figs += [boxplot(cv_data, passed_regs, metric, n_vizualized, metric_help)]
    for k in range(len(figs)):
        figs[k].savefig(f'fig_{k}.png',bbox_inches='tight')
    pass


if __name__ == "__main__":
    paramdict = {'datapath': 'AutoML/InsuranceData/insurance.csv',
                'n_regressors': 8,
                'metric_list': ['neg_mean_squared_error','neg_mean_absolute_error','r2'],
                'n_vizualized': 5,
                #GENERAL FORM of metric_help: { 'metric': [ higher score is better?, positive or negative score values, accociated stat function ] } 
                'metric_help': {'explained_variance': [True, 1, metrics.explained_variance_score], 'max_error': [False, 1, metrics.max_error],
                                'neg_mean_absolute_error': [False, -1, metrics.mean_absolute_error], 'neg_mean_squared_error': [False, -1, metrics.mean_squared_error],
                                'neg_root_mean_squared_error': [False, -1, metrics.mean_squared_error], 'neg_mean_squared_log_error': [False, -1, metrics.mean_squared_log_error],
                                'neg_median_absolute_error': [False, -1, metrics.median_absolute_error], 'r2': [True, 1, metrics.r2_score],
                                'neg_mean_poisson_deviance': [False, -1, metrics.mean_poisson_deviance], 'neg_mean_gamma_deviance': [False, -1, metrics.mean_gamma_deviance],
                                'neg_mean_absolute_percentage_error': [False, -1, metrics.mean_absolute_percentage_error], 'd2_absolute_error_score': [True, 1, metrics.d2_absolute_error_score],
                                'd2_pinball_score': [True, 1, metrics.d2_pinball_score], 'd2_tweedie_score': [True, 1, metrics.d2_tweedie_score]
                                },
                'score_method': 'neg_root_mean_squared_error'
                }

    comparison(**paramdict)
