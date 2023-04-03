import numpy as np
import pandas as pd

data = pd.read_csv('BetaDataExper/BigOTest/test_data/conductivity.csv')
print(data.values[:5,:5])
col_stds = list(data.std(axis=0))
data_copy2 = data.copy(deep=True)
data_copy3 = data.copy(deep=True)
data_copy4 = data.copy(deep=True)
data_copy5 = data.copy(deep=True)
nrows, ncols = data.shape
print(data.shape)

df_list = []
for copy in [data, data_copy2, data_copy3, data_copy4, data_copy5]:
    noise = np.zeros(shape=(nrows, ncols))
    for std, index in zip(col_stds, range(ncols)):
        noise[:,index] = np.reshape(np.random.normal(0, std, (nrows,1)),newshape=(nrows,))

    df_list += [pd.DataFrame(copy.values + noise)]

final = pd.concat(df_list)

final.to_csv('BetaDataExper/BigOTest/test_data/conductivity_exp.csv')

