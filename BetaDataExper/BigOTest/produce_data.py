import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import time

def prod_data(rows, cols):
    data = np.random.normal(loc=0, scale=1, size=(rows, cols))
    df = pd.DataFrame(data)
    with open(Path.cwd() / 'big_o_data.pkl', 'wb') as f:
        pickle.dump(df, f)
    return

def get_data():
    with open(Path.cwd() / 'big_o_data.pkl', 'rb') as f:
        df = pickle.load(f)
    print(df.head(3))
    return

if __name__ == '__main__':


    start_time = time.time()
    prod_data(1000000, 10)
            #  '  '  '
    get_data()
    print(f"--- {time.time() - start_time} seconds ---")
# data = pd.read_csv('BetaDataExper/BigOTest/test_data/conductivity.csv')
# print(data.values[:5,:5])
# col_stds = list(data.std(axis=0))
# data_copy2 = data.copy(deep=True)
# data_copy3 = data.copy(deep=True)
# data_copy4 = data.copy(deep=True)
# data_copy5 = data.copy(deep=True)
# nrows, ncols = data.shape
# print(data.shape)

# df_list = []
# for copy in [data, data_copy2, data_copy3, data_copy4, data_copy5]:
#     noise = np.zeros(shape=(nrows, ncols))
#     for std, index in zip(col_stds, range(ncols)):
#         noise[:,index] = np.reshape(np.random.normal(0, std, (nrows,1)),newshape=(nrows,))

#     df_list += [pd.DataFrame(copy.values + noise)]

# final = pd.concat(df_list)

# final.to_csv('BetaDataExper/BigOTest/test_data/conductivity_exp.csv')

