import pandas as pd
import numpy as np
import sys

x,y = sys.float_info.max,sys.float_info.min
matrix1 = pd.DataFrame(np.array([[x,y],[-x,-y]]))
matrix2 = pd.DataFrame(np.array([[x,x],[-x,-x]]))
matrix3 = pd.DataFrame(np.array([[x,x],[y,y],[-x,-x]]))
for i,j in zip([matrix1, matrix2, matrix3],['inf_small_slope','max_points','max_points_plus_one']):
    i.to_csv(f'BetaDataExper/MaxMinPoints/data/{j}.csv', header=False)