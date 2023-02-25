from utils import *

paramdict = {'datapath': 'AutoML\ConcreteData\Concrete_Data.csv',
            'which_regressors': {'ARDRegression': 1, 'AdaBoostRegressor': 1, 'BaggingRegressor': 1, 'BayesianRidge': 1, 'CCA': 1, 
                                 'DecisionTreeRegressor': 1, 'DummyRegressor': 1, 'ElasticNet': 1, 'ExtraTreeRegressor': 1, 
                                 'ExtraTreesRegressor': 1, 'GammaRegressor': 1, 'GaussianProcessRegressor': 1, 'GradientBoostingRegressor': 1, 
                                 'HistGradientBoostingRegressor': 1, 'HuberRegressor': 1, 'IsotonicRegression': 1, 'KNeighborsRegressor': 1, 
                                 'KernelRidge': 1, 'Lars': 1, 'Lasso': 1, 'LassoLars': 1, 'LassoLarsIC': 1, 'LinearRegression': 1, 
                                 'LinearSVR': 1, 'MLPRegressor': 1, 'MultiTaskElasticNet': 1, 'MultiTaskLasso': 1, 'NuSVR': 1, 
                                 'OrthogonalMatchingPursuit': 1, 'PLSCanonical': 1, 'PLSRegression': 1, 'PassiveAggressiveRegressor': 1, 
                                 'PoissonRegressor': 1, 'QuantileRegressor': 1, 'RANSACRegressor': 0, 'RadiusNeighborsRegressor': 1, 
                                 'RandomForestRegressor': 1, 'Ridge': 1, 'SGDRegressor': 1, 'SVR': 1, 'TheilSenRegressor': 1, 
                                 'TransformedTargetRegressor': 1, 'TweedieRegressor': 1
                                 },
            'metric_list': ['Mean Squared Error','Mean Absolute Error','R-Squared', 'Root Mean Squared Error'],
            'styledict': {'boxprops': {'linestyle': '-', 'linewidth': 1, 'color': 'black'},
                          'flierprops': {'marker': 'D', 'markerfacecolor': 'white', 'markersize': 4, 'linestyle': 'none'},
                          'medianprops': {'linestyle': '-.', 'linewidth': 1, 'color': 'black'},
                          'whiskerprops': {'linestyle': '--', 'linewidth': 1, 'color': 'black'},
                          'capprops': {'linewidth': 1, 'color': 'black'}, 'boxfill': 'lightgray', 'grid': True, 'dpi': 300.0 
                            },
            'n_vizualized_bp': 20,
            'n_vizualized_tb': 10,
            'test_set_size': 0.2,
            'n_cv_folds': 10,
            'score_method': 'Root Mean Squared Error'
            }


### Regular run ###
if __name__=="__main__":
    start = perf_counter()
    print(comparison(**paramdict))
    stop = perf_counter()

    print(f"Total time to execute: {stop - start:.2f}s")
