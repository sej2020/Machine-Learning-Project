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

#Load and Describe Data**************************************************************************************************************************
def load_pp_data():
    csv_path = os.path.abspath("PowerPlantData\CCPP\Folds5x2_pp.csv")
    print(csv_path)
    return pd.read_csv(csv_path)

pp = load_pp_data()
print(pp.describe())

#Import All Regressions**************************************************************************************************************************
estimators = all_estimators(type_filter='regressor')

all_regs = []
for name, RegressorClass in estimators:
    try:
        if name != 'DummyRegressor' and name != 'GaussianProcessRegressor':
            print('Appending', name)
            reg = RegressorClass()
            all_regs.append(reg)
    except Exception as e:
        print(e)

print(all_regs)

#Train/Test Split and Prepare Data*************************************************************************************************************
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

#Simultaneous Run********************************************************************************************************************
def run(model):
    print(f"checking {model}")
    try:
        cv_outer = KFold(n_splits=10, shuffle=True, random_state=2)
        cv_output_dict = cross_validate(model, pptrain_attrib, pptrain_labels, scoring=["neg_mean_squared_error","neg_mean_absolute_error","r2"], cv=cv_outer, return_estimator=True)
        return cv_output_dict
    except:
        pass

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
    return vizualize(cv_data, passed_models), test_best(cv_data)

def vizualize(cv_data, modellst):
    return box_rmse(cv_data, modellst), box_r2(cv_data, modellst), box_mae(cv_data, modellst), runtime(cv_data, modellst)

def runtime(cv_data, modellst):
    timefig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,modellst):
        df[j] = list(i[('fit_time')])
    sorted_index = df.median().sort_values(ascending=False).index
    df_sorted=df[sorted_index]
    df_sorted.boxplot(vert=False,grid=False)
    plt.xlabel('Run Time')
    plt.ylabel('Models')
    return timefig


def box_rmse(cv_data, modellst):
    RMSEfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,modellst):
        df[j] = list(np.sqrt(i['test_neg_mean_squared_error']*-1))
    sorted_index = df.median().sort_values(ascending=False).index
    df_sorted=df[sorted_index]
    df_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV Root Mean Squared Error (Lower is better)')
    return RMSEfig

def box_r2(cv_data, modellst):
    R2fig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,modellst):
        df[j] = list(i['test_r2'])
    sorted_index = df.median().sort_values().index
    df_sorted=df[sorted_index]
    df_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV R-Squared Score (Higher is better)')
    return R2fig

def box_mae(cv_data, modellst):
    MAEfig = plt.figure(constrained_layout=True)
    df = pd.DataFrame()
    for i,j in zip(cv_data,modellst):
        df[j] = list(i['test_neg_mean_absolute_error']*-1)
    sorted_index = df.median().sort_values(ascending=False).index
    df_sorted=df[sorted_index]
    df_sorted.boxplot(vert=False,grid=False)
    plt.xlabel(f'CV Mean Absolute Error (Lower is better)')
    return MAEfig

def test_best(cv_data):
    models = []
    rmse = []
    r2 = []
    mae = []
    for i in cv_data:
        x = list((np.sqrt(i['test_neg_mean_squared_error']*-1)))
        y = list(i['estimator'])
        for j in range(len(x)):
            if x[j] == min(x):
                best = y[j]
                print(best)
        predictions = best.predict(pptest_attrib)
        rmse += [np.sqrt(mean_squared_error(pptest_labels,predictions))]
        r2 += [r2_score(pptest_labels,predictions)]
        mae += [mean_absolute_error(pptest_labels,predictions)]
        models += [best]
        print(models)
    columnnames = ['rmse','r2','mae']
    df = pd.DataFrame(np.array([rmse,r2,mae]).T,index=models,columns=columnnames)
    return df
        

y = all_regs[0:20]
x = all_regs[0:3]
print(comparison(x))
plt.show()