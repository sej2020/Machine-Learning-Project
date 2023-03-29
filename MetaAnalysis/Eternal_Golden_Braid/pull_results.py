from yaml import load, dump, SafeLoader, SafeDumper
import numpy as np
from pathlib import Path

def main():
    input_path = Path("BetaDataExper/pipeline/outputs/")
    output_path = Path("MetaAnalysis/Eternal_Golden_Braid/")
    for i, output_folder in enumerate(input_path.glob("*")):
        results_path = output_folder / "results.yaml"
        metadata_path = output_folder / "metadata.yaml"
        with open(results_path, "r") as f:
            data = load(f, SafeLoader)
            
        with open(metadata_path, "r") as f:
            metadata = load(f, SafeLoader)
            
        for model in data:
            del data[model]["y_pred"]
            
        with open(output_path / "outputs" / f"data_{i}.yaml", "w") as f:
            dump(data, f, SafeDumper)
            
        with open(output_path / "outputs" / f"metadata_{i}.yaml", "w") as f:
            dump(metadata, f, SafeDumper)
    
    
if __name__=="__main__":
    main()