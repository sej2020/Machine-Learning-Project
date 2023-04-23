import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

# Load data
def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def main(memory_path, actual_runtime_path, theoretical_runtime_path, experiment):
    # Load data
    mem_df = load_data(memory_path)
    act_rt_df = load_data(actual_runtime_path)
    theo_rt_df = load_data(theoretical_runtime_path)
    row_counts = list(mem_df.iloc[:,0])
    print(act_rt_df)
    print(row_counts)
    pass
    # Setup for plotting
    label_dict_rt ={
        "tf-necd": "TensorFlow (NE-CD)",
        "tf-cod": "TensorFlow (COD)",
        "pytorch-qrcp": "PyTorch (QRCP)",
        "pytorch-qr": "PyTorch (QR)",
        "pytorch-svd": "PyTorch (SVD)",
        "pytorch-svddc": "PyTorch (SVDDC)",
        "sklearn-svddc": "scikit-learn (SVDDC)",
    }

    label_dict_mem ={
        "tf-necd": "TensorFlow (NE-CD)",
        "tf-cod": "TensorFlow (COD)",
        "pytorch-qrcp": "All PyTorch solvers",
        "sklearn-svddc": "scikit-learn (SVDDC)",
    }

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

    # Plotting memory
    solvers = list(mem_df.columns[1:])
    fig, ax = plt.subplots()
    for solver in solvers:
        if solver in ['tf-necd', 'tf-cod', 'pytorch-qrcp','sklearn-svddc']:
            ax.plot(row_counts, mem_df[solver], label=label_dict_mem[solver])
    ax.set_xlabel("Number of rows in dataset")
    ax.set_ylabel("Memory usage (GB)")
    ax.set_title("Memory usage of OLS solvers")
    labels = ["0", "25", "50", "75", "100", "125", "150", "175", "200", "225", "250"]
    ax.yaxis.set_major_locator(ticker.FixedLocator([i*0.25*10**11 for i in range(11)]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    plt.xscale("log")
    ax.legend()
    plt.savefig(f"BetaDataExper/BigOTest/postprocessing/mem_figs/{experiment}/all_memory.png", dpi=300)
    plt.clf()

    #Plotting small-scale memory
    solvers = list(mem_df.columns[1:])
    mem_df.drop(mem_df.tail(6).index, inplace = True)
    fig, ax = plt.subplots()
    for solver in solvers:
        if solver in ['tf-necd', 'tf-cod', 'pytorch-qrcp','sklearn-svddc']:
            ax.plot(row_counts[:-6], mem_df[solver], label=label_dict_mem[solver])
    ax.set_xlabel("Number of rows in dataset")
    ax.set_ylabel("Memory usage (MB)")
    ax.set_title("Memory usage of OLS solvers")
    labels = ["0", "25", "50", "75", "100", "125", "150", "175", "200", "225", "250"]
    ax.yaxis.set_major_locator(ticker.FixedLocator([i*0.25*10**8 for i in range(11)]))
    ax.yaxis.set_major_formatter(ticker.FixedFormatter(labels))
    plt.xscale("log")
    ax.legend()
    plt.savefig(f"BetaDataExper/BigOTest/postprocessing/mem_figs/{experiment}/small-scale_memory.png", dpi=300)
    plt.clf()


    # Plotting runtime
    act_rt_df_s = act_rt_df.iloc[:,1:].div(1e9)
    theo_rt_df_s = theo_rt_df.iloc[:,1:].div(1e9)

    solvers = list(act_rt_df_s.columns)
    for solver, color in zip(solvers,["red", "darkblue", "darkgreen", "orange", "purple", "mediumvioletred", "slategray"]):
        fig, ax = plt.subplots()
        ax.plot(row_counts, act_rt_df_s[solver], label=label_dict_rt[solver]+" - Actual", color=color)
        ax.plot(row_counts, theo_rt_df_s[solver], label=label_dict_rt[solver]+" - Theoretical", color=color, linestyle="dashed")
        ax.set_xlabel("Number of rows in dataset")
        ax.set_ylabel("Runtime (s)")
        ax.set_title("Runtime of OLS solvers")
        plt.xscale("log")
        ax.legend()
        plt.savefig(f"BetaDataExper/BigOTest/postprocessing/rt_figs/{experiment}/{solver}.png", dpi=300)
        plt.clf()

    # Plotting runtime - log
    act_rt_df_ms = act_rt_df.iloc[:,1:].div(1e6)
    theo_rt_df_ms = theo_rt_df.iloc[:,1:].div(1e6)

    solvers = list(act_rt_df_ms.columns)
    for solver, color in zip(solvers,["red", "darkblue", "darkgreen", "orange", "purple", "mediumvioletred", "slategray"]):
        fig, ax = plt.subplots()
        ax.plot(row_counts, act_rt_df_ms[solver], label=label_dict_rt[solver]+" - Actual", color=color)
        ax.plot(row_counts, theo_rt_df_ms[solver], label=label_dict_rt[solver]+" - Theoretical", color=color, linestyle="dashed")
        ax.set_xlabel("Number of rows in dataset")
        ax.set_ylabel("Runtime (ms)")
        ax.set_title("Runtime of OLS solvers")
        plt.xscale("log")
        plt.yscale("log")
        ax.legend()
        plt.savefig(f"BetaDataExper/BigOTest/postprocessing/rt_figs/{experiment}/{solver}_log.png", dpi=300)
        plt.clf()




if __name__ == '__main__':
    mem_path = "BetaDataExper/BigOTest/postprocessing/processed_output/bytes.csv"
    act_rt_path = "BetaDataExper/BigOTest/postprocessing/processed_output/actual_runtime.csv"
    the_rt_path = "BetaDataExper/BigOTest/postprocessing/processed_output/theoretical_runtime.csv"
    experiment = "Quartz" # "Quartz" or "Carbonate" or "Mac" or "RaspberryPi"
    main(memory_path=mem_path, actual_runtime_path=act_rt_path, theoretical_runtime_path=the_rt_path, experiment=experiment)
