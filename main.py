import pyabf
from utils import process, process_multiple, ExtractedAbf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def save_csv(res: ExtractedAbf, path: Path):
    with open(path, 'w') as f:
        f.write("\n".join(f"{x},{y}" for x, y in zip(res.x, res.y_avg)))


def extract_ages(path: Path):
    age_injection, age_patch = str(path.parents[-1]).split('_')
    return {
        "path": path,
        "age_injection": age_injection,
        "age_patch": age_patch
    }


if __name__ == "__main__":
    p = Path('./data/').glob('**/*.abf')
    files = [f for f in p if f.is_file()]
    file_dicts = []

    for file in files:
        plt.clf()
        fig, graph = plt.subplots(1)
        res_500 = process(pyabf.ABF(file), graph)
        plt.savefig(file.parent / (file.stem.split('.')[0] + '_500ms.png'))

        plt.clf()
        fig, graph = plt.subplots(1)
        res_root = process(pyabf.ABF(file), graph, from_root=True)
        plt.savefig(file.parent / (file.stem.split('.')[0] + '_crossing.png'))

        save_csv(res_root, file.parent / (file.stem.split('.')[0] + '_data.csv'))
        file_dicts.append({"area_root": res_root.area, "area_500ms": res_500.area, **extract_ages(file)})

    pd.DataFrame(file_dicts).to_csv('./data/output.csv', header=True)
