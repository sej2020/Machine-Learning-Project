from yaml import load, dump, SafeLoader, SafeDumper
import numpy as np
import pandas as pd
from pathlib import Path

def main(in_path, out_path, metric_to_use):
    df = pd.DataFrame(columns=["location_x", "location_y", "rotation", "major_axis_length", "minor_axis_length", metric_to_use])
    with open(in_path, "r") as f:
        data_viz_dict = load(f, SafeLoader)  
        for loc_rot, metric_dict in data_viz_dict.items():
            for metric, regr_dict in metric_dict.items():
                for regr, ax_dict in regr_dict.items():
                    for axes, error in ax_dict.items():
                        if metric == metric_to_use and regr == "sklearn-svddc":
                            location = loc_rot.split("_")[0].split("-")[1]
                            loc_x, loc_y = int(location.split(",")[0].strip("() ")), int(location.split(",")[1].strip("() "))
                            rotation = int(loc_rot.split("_")[1].split("-")[1]) % 180
                            maj_min_ax = axes.split("-")[1]
                            major_axis, minor_axis = int(maj_min_ax.split(",")[0].strip("[] ")), int(maj_min_ax.split(",")[1].strip("[] "))
                            df.loc[len(df.index)] = [loc_x, loc_y, rotation, major_axis, minor_axis, error]
        f.close()
    df.to_csv(out_path, index=False)
    print(df.head(5))
    print(df.shape)

                    

if __name__ == "__main__":
    
    for metric in ["MAE", "MSE", "RMSE", "R2", "elapsed_time"]:
        in_path = "BetaDataExper/HyperEllipsoid/big_sim_viz/big_hyp_sim.yaml"
        out_path = f"BetaDataExper/HyperEllipsoid/closed_form_figs/pred_error_{metric}.csv"
        
        main(in_path, out_path, metric)