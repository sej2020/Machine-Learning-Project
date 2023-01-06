import os
import sys
import cProfile
import pstats
import io
sys.path.append(os.getcwd())
from utils import *

paramdict = {'datapath': 'AutoML/EmployeeData/employees.csv',
            'n_regressors': -1,
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

### Regular run ###
start = perf_counter()
comparison(**paramdict)
stop = perf_counter()

print(f"Total time to execute: {stop - start:.2f}s")
###################

### Profiling run ###
# pr = cProfile.Profile()
# pr.enable()

# my_result = comparison(**paramdict)

# pr.disable()
# s = io.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
# ps.print_stats()

# with open("AutoML/ScikitLearn/parSim_v5/profile_report.prof", 'w') as f:
#     f.write(s.getvalue())
#####################