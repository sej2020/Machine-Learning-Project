import os
import pyaml
import sys
sys.path.append(os.getcwd())
from quartz_utils import *

def dump_to_yaml(path, object):
    with open(path, "w") as f_log:
        dump = pyaml.dump(object)
        f_log.write(dump)

paramdict = {'datapath': 'AutoML/EmployeeData/employees.csv',
            'n_regressors': 5,
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

runtimes = {}

worker_range = range(1, 8)
for n_workers in worker_range:
    times = comparison(**{**paramdict, "n_workers":n_workers}) # times will be returned as (total, regressor_time) in seconds
    runtimes[n_workers] = times
    
dump_to_yaml("sam_sim_output.yaml", runtimes)

print("Stuff was yeeted to proper location (on fleek)")