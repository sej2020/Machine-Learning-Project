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

print(all_regs)
print(all_reg_names)

def load_data(datapath):
    csv_path = os.path.abspath(datapath)
    return pd.read_csv(csv_path)

def find_strat_label(data):
    datarscore = data.corr()
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

def create_strat_cat():
    strat_label = str(find_strat_label(pp))
    description = pp.describe()
    print(description)
    strat_bins = list(description.loc['min':'max',strat_label])
    strat_bins[0], strat_bins[-1] = -np.inf, np.inf

    print(strat_bins)
    pp[f"{strat_label}_cat"] = pd.cut(pp[strat_label],bins=strat_bins,labels=[1,2,3,4])
    return pp

def data_split():
    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
    for train_index, test_index in split.split(pp,pp[f"{strat_label}_cat"]):
        train_set = pp.loc[train_index]
        test_set = pp.loc[test_index]
    for set_ in(train_set,test_set):
        set_.drop(f"{strat_label}_cat",axis=1,inplace=True)
    pptrain = train_set.copy()
    pptest = test_set.copy()

    pp_class = pptrain.columns[-1]
    print(pp_class)
    pptrain_attrib = pptrain.drop(pp_class,axis=1)
    pptrain_labels = pptrain[pp_class].copy()
    pptest_attrib = pptest.drop(pp_class,axis=1)
    pptest_labels = pptest[pp_class].copy()

    return pptrain_attrib, pptrain_labels, pptest_attrib, pptest_labels

def scale():
    scaler = StandardScaler()
    pptrain_attrib = scaler.fit_transform(pptrain_attrib)
    pptest_attrib = scaler.fit_transform(pptest_attrib)


num_pipeline = Pipeline([()])
#mother function that runs the models and returns several figures showing comparison of the models
def comparison(datapath, n_models, metric_list, n_vizualized):
    models = all_regs[0:n_models]
    model_names = all_reg_names[0:n_models]
    data = load_data(datapath)
    cv_data = []
    errors = []
    passed_models = []
    if 'neg_mean_squared_error' not in metric_list:
        metric_list = metric_list+['neg_mean_squared_error']
    for i in range(len(models)):
        x = run(models[i],metric_list)
        if type(x) == dict:
            cv_data += [x]
        else:
            errors += [models[i]]
    for j in range(len(models)):
        if models[j] not in errors:
            passed_models += [model_names[j]]
    figs = [test_best(cv_data, passed_models, metric_list)]
    for metric in metric_list:
        figs += [boxplot(cv_data, passed_models,metric,show)]
    for k in range(len(figs)):
        figs[k].savefig(f'fig_{k}.png',bbox_inches='tight')
    return test_best(cv_data, passed_models,metric_list)

#the function that performs cross-validation
def run(model,metric_list):
    print(f"checking {model}")
    try:
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=2)
        cv_output_dict = cross_validate(model, pptrain_attrib, pptrain_labels, scoring=metric_list, cv=cv_outer, return_estimator=True)
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
def test_best(cv_data, passed_models, metric_list):
    metric_columns = []
    for metric in metric_list:
        metric_columns += [[metric,[]]]
    for i in cv_data:
        x = list((np.sqrt(i['test_neg_mean_squared_error']*-1)))
        y = list(i['estimator'])
        for j in range(len(x)):
            if x[j] == min(x):
                best = y[j]
        predictions = best.predict(pptest_attrib)
        for k in metric_columns:
            #next line won't work. need to figure out how to use statistics module to calculate metrics on test predictions
            k[1] += [round(r2_score(pptest_labels,predictions),4)]
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


y = all_regs
y_names = all_reg_names
n_models = 5
x = all_regs[0:n_models]
x_names = all_reg_names[0:n_models]

paramdict = {'datapath': 'AutoML/PowerPlantData/Folds5x2_pp.csv',
            'n_models': 5,
            'metric_list': ["neg_mean_squared_error","neg_mean_absolute_error","r2"],
            'n_vizualized': 3,
    }
comparison(**paramdict)

