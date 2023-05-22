from utils import *

paramdict = {'id': 30,
            'datapath': 'AutoML/PowerPlantData/Folds5x2_pp.csv', # ConductivityData/train.csv'
            'which_regressors': {'ARDRegression': 0, 'AdaBoostRegressor': 0, 'BaggingRegressor': 0, 'BayesianRidge': 0, 'CCA': 0, 
                                 'DecisionTreeRegressor': 1, 'DummyRegressor': 0, 'ElasticNet': 1, 'ExtraTreeRegressor': 0, 
                                 'ExtraTreesRegressor': 0, 'GammaRegressor': 1, 'GaussianProcessRegressor': 0, 'GradientBoostingRegressor': 0, 
                                 'HistGradientBoostingRegressor': 0, 'HuberRegressor': 0, 'IsotonicRegression': 0, 'KNeighborsRegressor': 1, 
                                 'KernelRidge': 0, 'Lars': 1, 'Lasso': 1, 'LassoLars': 1, 'LassoLarsIC': 1, 'LinearRegression': 1, 
                                 'LinearSVR': 0, 'MLPRegressor': 0, 'MultiTaskElasticNet': 0, 'MultiTaskLasso': 0, 'NuSVR': 0, 
                                 'OrthogonalMatchingPursuit': 0, 'PLSCanonical': 0, 'PLSRegression': 0, 'PassiveAggressiveRegressor': 0, 
                                 'PoissonRegressor': 0, 'QuantileRegressor': 0, 'RANSACRegressor': 0, 'RadiusNeighborsRegressor': 0, 
                                 'RandomForestRegressor': 0, 'Ridge': 1, 'SGDRegressor': 0, 'SVR': 0, 'TheilSenRegressor': 0, 
                                 'TransformedTargetRegressor': 0, 'TweedieRegressor': 0
                                 },
            'metric_list': ['Mean Squared Error','Mean Absolute Error','R-Squared', 'Root Mean Squared Error'],
            'styledict': {'boxprops': {'linestyle': '-', 'linewidth': 1, 'color': 'black'},
                          'flierprops': {'marker': 'D', 'markerfacecolor': 'white', 'markersize': 4, 'linestyle': 'none'},
                          'medianprops': {'linestyle': '-.', 'linewidth': 1, 'color': 'black'},
                          'whiskerprops': {'linestyle': '--', 'linewidth': 1, 'color': 'black'},
                          'capprops': {'linewidth': 1, 'color': 'black'}, 'boxfill': 'lightgray', 'grid': True, 'dpi': 300.0 
                            },
            'figure_lst': ['Test_Best_Models'],
            'n_vizualized_bp': 20,
            'n_vizualized_tb': 10,
            'test_set_size': 0.2,
            'n_cv_folds': 10,
            'score_method': 'Root Mean Squared Error',
            'n_workers': 1,
            }


### Regular run ###
if __name__=="__main__":
    start = perf_counter()
    print(comparison_wrapper(2,paramdict)) # 'AutoML\PowerPlantData\Folds5x2_pp.csv' 'AutoML/ConcreteData/Concrete_Data.csv'
    # print(comparison(**paramdict))
    stop = perf_counter()

    print(f"Total time to execute: {stop - start:.2f}s")
