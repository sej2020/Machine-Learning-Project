from yaml import load, dump, SafeLoader, SafeDumper
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def construct_info_dict():
    input_path = Path("MetaAnalysis\Eternal_Golden_Braid\outputs")

    data_for_viz = [[] for i in range(50)] # [ [{model_1: MSE, model_2: MSE,...}, metadata_0],... ]
              #            from data_0

    for i, yaml_file in enumerate(input_path.glob("*")):

        with open(yaml_file, "r") as f:
            results_dict = load(f, SafeLoader)
            file_path = str(yaml_file)
            file_name = file_path.rsplit("\\", 1)[1]
            info = [file_name.split("_")[0]] + [int(file_name.split("_")[1].split(".")[0])]
            match info:
                case ['data', index]:
                    data_for_viz[index] += [{model: results_dict[model]['MSE'] for model in results_dict}]
                case['metadata', index]:
                    data_for_viz[index] += [results_dict['input_data']]

    data_for_viz_dict = {}
    for i,j in data_for_viz:
        data_for_viz_dict[j] = i

        
    output_path = Path("MetaAnalysis/Eternal_Golden_Braid/")
    with open(output_path / "data_for_viz.yaml", "w") as f:
        dump(data_for_viz_dict, f, SafeDumper)

    pass


def create_viz(rotation='0'):
     
    with open("MetaAnalysis\Eternal_Golden_Braid\data_for_viz.yaml", "r") as f:
        data_viz_dict = load(f, SafeLoader)
        relevant_results = {}
        for k,v in data_viz_dict.items():
            if k.split('-')[1] == f'dim_3drot_{rotation}.csv':
                relevant_results[k] = v

    dimensions = [3, 5, 10, 50, 100, 250, 500, 1000, 5000, 10000]
    sorted_index = sorted(relevant_results, key=lambda x: int(x.split('-')[0].split('_')[1]), reverse=False)
    mxnet_scores = [relevant_results[index]['mxnet_lst_sq'] for index in sorted_index] 
    pytorch_scores = [relevant_results[index]['pytorch-lst_sq'] for index in sorted_index]
    sklearn_scores = [relevant_results[index]['sklearn-lst_sq'] for index in sorted_index]
    tf_fast_scores = [relevant_results[index]['tf-lst_sq_fast'] for index in sorted_index]
    tf_slow_scores = [relevant_results[index]['tf-lst_sq_slow'] for index in sorted_index]

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
    # plt.ticklabel_format(style = 'plain')

    plt.plot(dimensions, mxnet_scores, label="MXNet")
    plt.plot(dimensions, pytorch_scores, label="PyTorch")
    plt.plot(dimensions, sklearn_scores, label="scikit-learn")
    plt.plot(dimensions, tf_fast_scores, label="TensorFlow-fast")
    plt.plot(dimensions, tf_slow_scores, label="TensorFlow-slow")
    plt.xscale("log")
    plt.legend()
    plt.yscale("log")
    if rotation > 0:
        plt.title(f"OLS Error over High-Dimensional Data\n({rotation}{chr(176)} rotation)")
    else:
        plt.title("OLS Error over High-Dimensional Data")
    plt.ylabel("MSE")
    plt.xlabel("Hyperellipsoid Dimensions")
    plt.savefig(f"MetaAnalysis\Eternal_Golden_Braid\OLS_over_high_dim_rot_{rotation}.png")
    plt.clf()

    return relevant_results


    
    
if __name__=="__main__":
    for rot in [0,5,15,30,90]:
        print(create_viz(rotation=rot))