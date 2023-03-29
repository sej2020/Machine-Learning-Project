from yaml import load, dump, SafeLoader, SafeDumper
import numpy as np
from pathlib import Path

def construct_info_dict():
    input_path = Path("MetaAnalysis\Eternal_Golden_Braid\outputs")

    data_for_viz = [[] for i in range(50)] # [ [{model_1: RMSE, model_2: RMSE,...}, metadata_0],... ]
              #            from data_0

    for i, yaml_file in enumerate(input_path.glob("*")):

        with open(yaml_file, "r") as f:
            results_dict = load(f, SafeLoader)
            file_path = str(yaml_file)
            file_name = file_path.rsplit("\\", 1)[1]
            info = [file_name.split("_")[0]] + [int(file_name.split("_")[1].split(".")[0])]
            match info:
                case ['data', index]:
                    data_for_viz[index] += [{model: results_dict[model]['RMSE'] for model in results_dict}]
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

    return relevant_results


    
    
if __name__=="__main__":
    print(create_viz())