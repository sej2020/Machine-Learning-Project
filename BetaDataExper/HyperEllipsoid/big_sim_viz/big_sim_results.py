from yaml import load, dump, SafeLoader, SafeDumper
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

# def construct_info_dict():
#     input_path = Path("BetaDataExper/pipeline/outputs")

#     info_dict = {}

#     for i, output_folder in enumerate(input_path.glob("*")):
#         results_path = output_folder / "results.yaml"
#         metadata_path = output_folder / "metadata.yaml"
#         with open(results_path, "r") as f:
#             results = load(f, SafeLoader)
            
#         with open(metadata_path, "r") as f:
#             metadata = load(f, SafeLoader)

#         hyp_data_str = metadata["input_data"]
#         splitby_ = hyp_data_str.split("_")
#         print(splitby_)
#         loc_rot_info = splitby_[1] + "_" + splitby_[3]
#         ax_ratio = splitby_[2]
#         print(loc_rot_info)

#         if loc_rot_info not in info_dict.keys():
#             info = {}
#             for measure in ["MAE", "MSE", "R2", "RMSE", "elapsed_time"]:
#                 info[measure] = {k: {ax_ratio: v[measure]} for k,v in results.items()}
#             info_dict[loc_rot_info] = info

#         else:
#             for k,v in info_dict[loc_rot_info].items():
#             # "MAE", dict{}
#                 for k1,v1 in v.items():
#                 # mxnet, dict{}
#                     v1[ax_ratio] = results[k1][k]

#     output_path = Path("BetaDataExper/HyperEllipsoid")
#     with open(output_path / "big_hyp_sim.yaml", "w") as f:
#         dump(info_dict, f, SafeDumper)
#     pass


def create_viz():
     
    reg_label_lookup = {
        "tf-necd": "TensorFlow (NE-CD)",
        "tf-cod": "TensorFlow (COD)",
        "pytorch-qrcp": "PyTorch (QRCP)",
        "pytorch-qr": "PyTorch (QR)",
        "pytorch-svd": "PyTorch (SVD)",
        "pytorch-svddc": "PyTorch (SVDDC)",
        "sklearn-svddc": "scikit-learn (SVDDC)",
        "mxnet-svddc": "MXNet (SVDDC)"
    } 

    metric_label_lookup = {"MAE": "MAE",
                           "MSE": "MSE",
                           "R2": "R-Squared",
                           "RMSE": "RMSE",
                           "elapsed_time": "Runtime"}
    
    def metric_y_label(metric, y_values):
        match metric:
            case "MAE": 
                return [0,12]
            case "MSE": 
                return [0,160]
            case "R2": 
                return [min(y_values)+0.1*min(y_values), max(y_values)+0.1*abs(max(y_values))]
            case "RMSE":
                return [0,15]
            case "elapsed_time":
                [10**(-4),10**(-5)]

    with open("BetaDataExper/HyperEllipsoid/big_hyp_sim.yaml", "r") as f:
        data_viz_dict = load(f, SafeLoader)

        SMALL_SIZE = 10
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 18
        CHONK_SIZE = 24

        plt.rcParams["font.family"] = "Times New Roman"
        plt.rc('axes', titlesize=BIGGER_SIZE, labelsize=MEDIUM_SIZE, facecolor="xkcd:black")
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE, facecolor="xkcd:white", edgecolor="xkcd:black") #  powder blue
            
        sns.set_style("whitegrid", {'font.family':['serif'], 'axes.edgecolor':'black','ytick.left': True})

        for loc_rot, metric_dict in data_viz_dict.items():
            for metric, regr_dict in metric_dict.items():
                fig, ax = plt.subplots()
                ax.set_title(f"{metric_label_lookup[metric]} for OLS Solvers as Axis Ratio Decreases", wrap=True)
                ax.set_ylabel(metric_label_lookup[metric])
                ax.set_xlabel("Major-Minor Axis Ratio")
                for regr, axes_dict in regr_dict.items():
                    y = list(axes_dict.values())
                    y = y[1:] + [y[0]]
                    ax.plot([i for i in range(10)], y, label=reg_label_lookup[regr])
                labels = ["10:1", "10:2", "10:3", "10:4", "10:5", "10:6", "10:7", "10:8", "10:9", "1:1"]
                ax.xaxis.set_major_locator(ticker.FixedLocator([i for i in range(10)]))
                ax.xaxis.set_major_formatter(ticker.FixedFormatter(labels))
                ax.set_ylim(metric_y_label(metric,y))
                if metric=="elapsed_time":
                    ax.legend()
                fig.savefig(f"BetaDataExper/HyperEllipsoid/big_sim_viz/{metric}/{loc_rot}_{metric}.jpg", dpi=300)
                plt.close(fig)
    pass


    
if __name__=="__main__":
    print(create_viz())

