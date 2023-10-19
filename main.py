import pyabf
from utils import process, process_multiple, ExtractedAbf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
from pathlib import Path


def save_csv(res: ExtractedAbf, path: Path):
    with open(path, 'w') as f:
        f.write("\n".join(f"{x},{y}" for x, y in zip(res.x, res.y_avg)))


def extract_ages(path: Path):
    age_injection, age_patch = path.parts[-2].split('_')
    return {
        "path": path,
        "age_injection": age_injection,
        "age_patch": age_patch
    }


if __name__ == "__main__":
    p = Path('./data/').glob('**/*.abf')
    files = [f for f in p if f.is_file()]
    file_dicts = []

    fig, graph = plt.subplots(1)

    print("Calculating per cell...")
    for file in tqdm(files):
        abf = pyabf.ABF(file)
        res_500 = process(abf, graph)
        plt.savefig(file.parent / (file.stem.split('.')[0] + '_500ms.png'))
        plt.cla()

        res_root = process(abf, graph, from_root=True)
        plt.savefig(file.parent / (file.stem.split('.')[0] + '_crossing.png'))
        plt.cla()

        save_csv(res_root, file.parent / (file.stem.split('.')[0] + '_data.csv'))
        file_dicts.append({"area_root": res_root.area, "area_500ms": res_500.area, **extract_ages(file)})

    print("Saving per-cell csv...")
    pd.DataFrame(file_dicts).to_csv('./data/per_cell_output.csv', header=True)

    print("Calculating per age group...")

    group_dicts = []
    groups = set([file.parent for file in files])
    for group in tqdm(groups):
        group_files = [file for file in files if str(group) in str(file)]

        res_500 = process_multiple(group_files, graph)
        plt.savefig(group / 'age_avg_500ms.png')
        plt.cla()

        res_root = process_multiple(group_files, graph, from_root=True)
        plt.savefig(group / 'age_avg_crossing.png')
        plt.cla()

        save_csv(res_root, group / 'age_avg.csv')
        group_dicts.append({"area_root": res_root.area, "area_500ms": res_500.area, **extract_ages(group / '123.abf')})

    print("Saving age group csv...")
    pd.DataFrame(group_dicts).to_csv('./data/per_age_output.csv', header=True)
