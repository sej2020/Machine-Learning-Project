import numpy as np
import pandas as pd

def main(outlier_loc):
    X = np.array([[i for i in range(100)] + [outlier_loc[0]]])
    y = np.array([[i/10 for i in range(100)] + [outlier_loc[1]]])
    df = pd.DataFrame(np.concatenate((X.T, y.T), axis=1))
    df.to_csv(f"BetaDataExper/outlier_data/data/outlier_loc-{outlier_loc}.csv", header=False, index=False)

for x in [-10, 0 , 5, 10, 20]:
    for y in [-100, -1000, 100, 1000]:
        outlier = (x,y)
        main(outlier)

