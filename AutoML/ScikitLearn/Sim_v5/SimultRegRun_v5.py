import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def get_all_regs():
    estimators = all_estimators(type_filter='regressor')
    forbidden_estimators = (
        "DummyRegressor", "GaussianProcessRegressor", 
        "QuantileRegressor", "SGDRegressor", 
        "MultiOutputRegressor", "RegressorChain",
        "StackingRegressor", "VotingRegressor"
        )

    all_regs = []
    all_reg_names = []
    for name, RegressorClass in estimators:
        try:
            if name not in forbidden_estimators:
                print('Appending', name)
                reg = RegressorClass()
                all_regs.append(reg)
                all_reg_names.append(name)
        except Exception as e:
            print(e)

    return all_regs, all_reg_names

def load_data(datapath):
    csv_path = os.path.abspath(datapath)
    return pd.read_csv(csv_path)

def find_strat_label(raw_data):
    datarscore = raw_data.corr()
    label = datarscore.columns[-1]
    datarscore = datarscore.drop(label)
    max_cor = datarscore[label].idxmax()
    min_cor = datarscore[label].idxmin()
    max_cor_val = datarscore[label].max()
    min_cor_val = datarscore[label].min()
    if abs(min_cor_val) > max_cor_val:
        return min_cor
    else:
        return max_cor

def create_strat_cat(raw_data,strat_label):
    description = raw_data.describe()
    strat_bins = list(description.loc['min':'max',strat_label])
    strat_bins[0], strat_bins[-1] = -np.inf, np.inf
    raw_data[f"{strat_label}_cat"] = pd.cut(raw_data[strat_label],bins=strat_bins,labels=[1,2,3,4])
    data_w_strat_cat = raw_data
    return data_w_strat_cat

def data_split(data, strat_label):
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(data,data[f"{strat_label}_cat"]):
        train_set = data.loc[train_index]
        test_set = data.loc[test_index]
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

def scale(train_attrib, train_labels, test_attrib, test_labels):
    scaler = StandardScaler()
    scaled_train_attrib = scaler.fit_transform(train_attrib)
    scaled_test_attrib = scaler.fit_transform(test_attrib)
    return scaled_train_attrib, train_labels, scaled_test_attrib, test_labels

def data_transform(datapath):
    raw_data = load_data(datapath)
    strat_label = find_strat_label(raw_data)
    data_w_strat_cat = create_strat_cat(raw_data,strat_label)
    split_data = data_split(data_w_strat_cat, strat_label)
    train_attrib, train_labels, test_attrib, test_labels = scale(*split_data)
    return train_attrib, train_labels, test_attrib, test_labels

#mother function that runs the models and returns several figures showing comparison of the models
def comparison(datapath, n_models, metric_list, n_vizualized):
    models = get_all_regs()[0][0:n_models]
    model_names = get_all_regs()[1][0:n_models]
    train_attrib, train_labels, test_attrib, test_labels = data_transform(datapath)
    cv_data = []
    errors = []
    passed_models = []
    if 'neg_mean_squared_error' not in metric_list:
        metric_list = metric_list+['neg_mean_squared_error']
    for i in range(len(models)):
        x = run(models[i], metric_list, train_attrib, train_labels)
        if type(x) == dict:
            cv_data += [x]
        else:
            errors += [models[i]]
    for j in range(len(models)):
        if models[j] not in errors:
            passed_models += [model_names[j]]
    figs = [test_best(cv_data, passed_models, metric_list, test_attrib, test_labels)]
    for metric in metric_list:
        figs += [boxplot(cv_data, passed_models,metric,n_vizualized)]
    for k in range(len(figs)):
        figs[k].savefig(f'fig_{k}.png',bbox_inches='tight')
    pass

#the function that performs cross-validation
def run(model, metric_list, train_attrib, train_labels):
    print(f"checking {model}")
    try:
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=2)
        cv_output_dict = cross_validate(model, train_attrib, train_labels, scoring=metric_list, cv=cv_outer, return_estimator=True)
        return cv_output_dict
    except:
        pass

#metric must be a string
#show is how many of top models are vizualized
def boxplot(cv_data, passed_models, metric, show):
    boxfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        if metric[:3] == 'neg':
            df[j] = list(i['test_'+metric]*-1)
        else:
            df[j] = list(i['test_'+metric])

    metric_dict = {"r2": False, "neg_mean_squared_error": True,"neg_mean_absolute_error": True}
    sorted_index = df.median().sort_values(ascending=not metric_dict[metric]).index
    df_sorted = df_sorted=df[sorted_index]
    df_sorted.iloc[:, :show].boxplot(vert=False,grid=False)
    plt.xlabel(f'CV {metric}')
    plt.ylabel('Models')
    return boxfig


#takes the model produced by the best cv run and runs it over the test data. returns table comparing model performance on test data
def test_best(cv_data, passed_models, metric_list, test_attrib, test_labels):
    metric_columns = []
    for metric in metric_list:
        metric_columns += [[metric,[]]]
    for i in cv_data:
        x = list((np.sqrt(i['test_neg_mean_squared_error']*-1)))
        y = list(i['estimator'])
        for j in range(len(x)):
            if x[j] == min(x):
                best = y[j]
        predictions = best.predict(test_attrib)
        for k in metric_columns:
            #next line won't work. need to figure out how to use statistics module to calculate metrics on test predictions
            k[1] += [round(r2_score(test_labels,predictions),4)]
    columnnames = metric_list
    final_columns = []
    for m in metric_columns:
        final_columns += [m[1]]
    df = pd.DataFrame(np.array(final_columns).T,index=passed_models,columns=columnnames)
    if metric_list[0] != 'r2':
        notr2 = True
    else:
        notr2 = False
    sorted_df = df.sort_values(by=metric_list[0],ascending=notr2)
    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=sorted_df.values, rowLabels=sorted_df.index, colLabels=sorted_df.columns, loc='center')
    fig.tight_layout()
    return fig


paramdict = {'datapath': 'AutoML/PowerPlantData/Folds5x2_pp.csv',
            'n_models': 5,
            'metric_list': ["neg_mean_squared_error","neg_mean_absolute_error","r2"],
            'n_vizualized': 3,
    }
comparison(**paramdict)