# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# **First 20 Models Simultaneous Run**
# 
# General Imports

# %%
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# Automatically Importing All Regressions

# %%

estimators = all_estimators(type_filter='regressor')

all_regs = []
for name, RegressorClass in estimators:
    try:
        if name != 'DummyRegressor' and name != 'GaussianProcessRegressor' and name != 'QuantileRegressor' and name != 'SGDRegressor':
            print('Appending', name)
            reg = RegressorClass()
            all_regs.append(reg)
    except Exception as e:
        print(e)

print(all_regs)

# %% [markdown]
# Load and Describe Data

# %%
def load_pp_data():
    csv_path = r"C:\Users\18123\OneDrive\Documents\IUBloomington\Machine-Learning-Project\PowerPlantData\CCPP\Folds5x2_pp.csv"
    return pd.read_csv(csv_path)

pp = load_pp_data()
print(pp.describe())

# %% [markdown]
# Train/Test Split and Preprocess Data

# %%
pp["AT_cat"] = pd.cut(pp["AT"],bins=[0.,10.,20.,30.,np.inf],labels=[1,2,3,4])

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(pp,pp["AT_cat"]):
    train_set = pp.loc[train_index]
    test_set = pp.loc[test_index]

for set_ in(train_set,test_set):
    set_.drop("AT_cat",axis=1,inplace=True)

pptrain = train_set.copy()
pptest = test_set.copy()

pptrain_attrib = pptrain.drop("PE",axis=1)
pptrain_labels = pptrain["PE"].copy()
pptest_attrib = pptest.drop("PE",axis=1)
pptest_labels = pptest["PE"].copy()

scaler = StandardScaler()
scaler.fit_transform(pptrain_attrib)

# %% [markdown]
# Simultaneous Run

# %%
def comparison(modellst):
    cv_data = []
    errors = []
    passed_models = []
    for i in range(len(modellst)):
        x = run(modellst[i])
        if type(x) == dict:
            cv_data += [x]
        else:
            errors += [i]
    for j in range(len(modellst)):
        if j not in errors:
            passed_models += [modellst[j]]
    display(test_best(cv_data, passed_models))
    return box_rmse(cv_data, passed_models), box_r2(cv_data, passed_models), box_mae(cv_data, passed_models), runtime(cv_data, passed_models)


def run(model):
    print(f"checking {model}")
    try:
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=2)
        cv_output_dict = cross_validate(model, pptrain_attrib, pptrain_labels, scoring=["neg_mean_squared_error","neg_mean_absolute_error","r2"], cv=cv_outer, return_estimator=True)
        return cv_output_dict
    except:
        pass


def runtime(cv_data, passed_models):
    timefig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        df[j] = list(i[('fit_time')])
    sorted_index = df.median().sort_values().index
    df_sorted=df[sorted_index]
    top20 = df_sorted.drop(columns=df_sorted.columns[20:])
    top20_sorted_index = top20.median().sort_values(ascending=False).index
    top20_sorted=top20[top20_sorted_index]
    top20_sorted.boxplot(vert=False,grid=False)
    plt.xlabel('Run Time')
    plt.ylabel('Models')
    return timefig


def box_rmse(cv_data, passed_models):
    RMSEfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        df[j] = list(np.sqrt(i['test_neg_mean_squared_error']*-1))
    sorted_index = df.median().sort_values().index
    df_sorted=df[sorted_index]
    top20 = df_sorted.drop(columns=df_sorted.columns[20:])
    top20_sorted_index = top20.median().sort_values(ascending=False).index
    top20_sorted=top20[top20_sorted_index]
    top20_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV Root Mean Squared Error (Lower is better)')
    return RMSEfig


def box_r2(cv_data, passed_models):
    R2fig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        df[j] = list(i['test_r2'])
    sorted_index = df.median().sort_values(ascending=False).index
    df_sorted=df[sorted_index]
    top20 = df_sorted.drop(columns=df_sorted.columns[20:])
    top20_sorted_index = top20.median().sort_values().index
    top20_sorted=top20[top20_sorted_index]
    top20_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV R-Squared Score (Higher is better)')
    return R2fig


def box_mae(cv_data, passed_models):
    MAEfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,passed_models):
        df[j] = list(i['test_neg_mean_absolute_error']*-1)
    sorted_index = df.median().sort_values().index
    df_sorted=df[sorted_index]
    top20 = df_sorted.drop(columns=df_sorted.columns[20:])
    top20_sorted_index = top20.median().sort_values(ascending=False).index
    top20_sorted=top20[top20_sorted_index]
    top20_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV Mean Absolute Error (Lower is better)')
    return MAEfig


def test_best(cv_data, passed_models):
    rmse = []
    r2 = []
    mae = []
    for i in cv_data:
        x = list((np.sqrt(i['test_neg_mean_squared_error']*-1)))
        y = list(i['estimator'])
        for j in range(len(x)):
            if x[j] == min(x):
                best = y[j]
        predictions = best.predict(pptest_attrib)
        rmse += [np.sqrt(mean_squared_error(pptest_labels,predictions))]
        r2 += [r2_score(pptest_labels,predictions)]
        mae += [mean_absolute_error(pptest_labels,predictions)]
    columnnames = ['rmse','r2','mae']
    df = pd.DataFrame(np.array([rmse,r2,mae]).T,index=passed_models,columns=columnnames)
    sorted_df = df.sort_values(by="rmse",ascending=True)
    final_df = sorted_df.style.format("{:.4f}")
    return final_df


y = all_regs
x = all_regs[0:8]
comparison(y)
plt.show()


