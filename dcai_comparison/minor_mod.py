import pandas as pd
from pathlib import Path

data = Path("dcai_comparison/csv_data/liver-disorders.csv")
df = pd.read_csv(data, index_col=None, header=None)
# drop last column of df

if len(df.columns) == 7:
    df = df.iloc[:, :-1]
    df.to_csv(data, index=False, header=False) 
