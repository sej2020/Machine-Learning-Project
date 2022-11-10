import os
import sys
sys.path.append(os.getcwd())
from v5_utils import *

paramdict = {'datapath': 'AutoML/PowerPlantData/Folds5x2_pp.csv',
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
