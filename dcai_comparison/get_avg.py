import pandas as pd
from pathlib import Path

dir = Path("dcai_comparison/comp_tests")
best = []

for file in dir.glob("ld_run*.csv"):
    df = pd.read_csv(file, index_col=0, header=0)
    best_rmse = df.iloc[0, 0]
    best.append(best_rmse)
    
print(f"Best RMSE: {min(best)}")
print(f"Mean RMSE: {sum(best)/len(best)}")
print