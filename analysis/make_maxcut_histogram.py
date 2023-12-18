import argparse
from collections import defaultdict
from mat73 import loadmat as mat73_loadmat
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import pickle
import re
from scipy.io import loadmat  # type: ignore
import seaborn as sns
import os

from IPython import embed


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--summary_df_fname", type=str, required=True, help="summary dataframe filename")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 18, "figure.figsize": (20, 8)})

    hparams = get_hparams()

    with open(hparams.summary_df_fname, "rb") as f:
        df = pickle.load(f)

    df["dataset name"] = [os.path.basename(x).split(".")[0] for x in df["data_path"]]
    df["warm-start"] = (df["warm_start_strategy"] != "none")
    df["temp"] = df["warm-start"].apply(lambda x: "warm" if x else "cold")
    df["solve_strategy"] = df[["solver", "temp"]].apply(lambda x: "/".join(x.astype(str)), axis=1)

    subset_df = df

    num_vertices_dict = defaultdict()
    nnz_dict = defaultdict()
    for data_path in subset_df["data_path"].unique():
        try:
            problem = loadmat(data_path)
            dict_format = False
        except:
            problem = mat73_loadmat(data_path)
            dict_format = True
        if "Gset" in data_path:
            C = problem["Problem"][0][0][1]
        elif "DIMACS" in data_path and not dict_format:
            C = problem["Problem"][0][0][2]
        elif "DIMACS" in data_path and dict_format:
            C = problem["Problem"]["A"]
        else:
            raise ValueError("Unknown path type")
        
        num_vertices_dict[data_path] = C.shape[0]
        nnz_dict[data_path] = C.nnz

    # sort the df the correct way
    data_paths = list(subset_df["data_path"].unique())
    data_paths.sort(key=lambda s: num_vertices_dict[s])

    dfs = []
    for data_path in data_paths:
        dfs.append(subset_df[subset_df["data_path"] == data_path].sort_values(
            by="solve_strategy", kind="quicksort"))

    subset_df = pd.concat(dfs).reset_index(drop=True)

    # make the figure
    palette = sns.color_palette()
    ax = sns.barplot(
        data=subset_df,
        x="dataset name",
        y="time (sec)",
        hue="solve_strategy",
        palette=[palette[0], palette[0], palette[1], palette[1]],
        width=0.8,
        errorbar=None,
        edgecolor='black',
        linewidth=2)

    for bars, hatch, legend_handle in zip(ax.containers, ['', '//', '', '//'], ax.legend_.legend_handles):
        for bar, color in zip(bars, palette):
            bar.set_hatch(hatch)
        # update the existing legend, use twice the hatching pattern to make it denser
        legend_handle.set_hatch(hatch + hatch)

    plt.xlabel("")
    plt.yscale("log")
    sns.despine()
    ax.set_ylim((1, 5000000))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
    ax.tick_params(width=2)
    plt.xticks(rotation=45)
    plt.setp(ax.spines.values(), linewidth=2)
    plt.tight_layout()
    plt.savefig("maxcut_warm-start_bar_chart.png")
    plt.clf()
    os.system("convert maxcut_warm-start_bar_chart.png -trim maxcut_warm-start_bar_chart.png")