from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy import MetaData
from sqlalchemy import Table, Column, Integer, String
from sqlalchemy import ForeignKey
from sqlalchemy import insert
from sqlalchemy import select
from sqlalchemy import func
from typing import List
from typing import Optional
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import relationship

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from csv import DictWriter
import time

#__________________________________________________________________________________
#----------------------------------------------------------------------------------
# SETTING UP DATBASE
#__________________________________________________________________________________
#----------------------------------------------------------------------------------
engine = create_engine("sqlite+pysqlite:///:memory:", echo=True)
metadata_obj = MetaData(bind=engine) #### brrrrrrrm brrrrrrrm

requests = Table("requests", metadata_obj, 
                    Column("id", Integer, primary_key=True), 
                    Column("metrics", String(30)),
                    Column("viz", Integer),
                    Column("folds", Integer),
                    Column("data_file", String),
                    Column("return_results", String)
                    )

metadata_obj.create_all(engine)

stmt = insert(requests).values(id=567, metrics="rmse, mse, mae", viz=1, folds=10, data_file='toydata.txt')
with engine.begin() as conn:
    result = conn.execute(stmt)
#__________________________________________________________________________________
#----------------------------------------------------------------------------------



def main(id):
    #s3_in_buck = S3Service('incoming_data')
    #s3_out_buck = S3Service('outgoing_data')
    paramdict = retrieve_params(id)
    out_file_name = comparison(**paramdict)
    # s3_out_buck.upload_file(out_file_name)
    update_db_w_results(out_file_name, id)

    time.sleep(10)
    path = pathlib.Path(out_file_name)
    path.unlink()
    #to see if update worked:
    main_table = metadata_obj.tables['requests']
    stmt = select(main_table).where(main_table.c.id == id)
    with engine.begin() as conn:
        results = conn.execute(stmt)
        for row in results:
            print(row)
    return results


def retrieve_params(id: int) -> dict:    
    main_table = metadata_obj.tables['requests']

    path = '/tempdata'
    paramdict = {}
    stmt = select(main_table).where(main_table.c.id == id)
    with engine.begin() as conn:
        single_row = conn.execute(stmt)
        for row in single_row:
            for k,v in zip(row.keys(), row):
                    paramdict[k] = v
    print(paramdict)
    file_name = paramdict['data_file']
    # s3_in_buck.download_file(file_name, path)
    f = open(os.path.join(os.path.dirname(__file__), file_name), 'r+')
    print(f.read())
    f.write('\nGotcha')
    f.close()
    # s3_in_buck.delete(file_name)
    paramdict['datapath'] = file_name
    del paramdict['data_file']
    return paramdict


def comparison(id, metrics, viz, folds, datapath, return_results):
    out_dict = {'Reg Name': [{'Same Reg Name': [100, 200, 300,'Reg Obj']}, {'Same Reg Name': [100, 200, 300,'Reg Obj']}, {'Same Reg Name': [100, 200, 300,'Reg Obj']}], 
                'Another Reg':[{'Same Reg Name': [101, 201, 301,'Reg Obj']}, {'Same Reg Name': [101, 201, 301,'Reg Obj']}, {'Same Reg Name': [101, 201, 301,'Reg Obj']}], 
                'Last One':[{'Same Reg Name': [102, 202, 302,'Reg Obj']}, {'Same Reg Name': [102, 202, 302,'Reg Obj']}, {'Same Reg Name': [102, 202, 302,'Reg Obj']}]
                }
    out_path = f"performance_stats_{id}.csv"
    metric_list = ['RMSE', 'MSE', 'MAE']
    write_results(out_path, out_dict, metric_list)
    return out_path


def update_db_w_results(result_data_name: str, id: int) -> None:
    main_table = metadata_obj.tables['requests']
    stmt = main_table.update().values(return_results = result_data_name).where(main_table.c.id == id)
    with engine.begin() as conn:
        conn.execute(stmt)


def write_results(path: str, data: dict, metrics: list) -> None:
    acc = {f"{regr}-{metric}" : [] for regr in data for metric in metrics}
    for regressor, runs in data.items():
        for fold, run in enumerate(runs):
            for metric_idx, value in enumerate(list(run.values())[0]):
                if metric_idx < len(metrics):
                    acc[f"{regressor}-{metrics[metric_idx]}"].append(value)
                    
    df = pd.DataFrame(acc)
    df.to_csv(path)



if __name__ == '__main__':
    #the panopticon
    main(567)