import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
import re
import seaborn as sns
import os

from IPython import embed


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--summary_df_fname", type=str, required=True, help="summary dataframe filename")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    plt.rcParams.update({"font.size": 18, "figure.figsize": (20, 6)})

    hparams = get_hparams()

    with open(hparams.summary_df_fname, "rb") as f:
        df = pickle.load(f)

    df["dataset name"] = [os.path.basename(x).split(".")[0] for x in df["data_path"]]
    df["warm-start"] = (df["warm_start_strategy"] != "none")
    df["temp"] = df["warm-start"].apply(lambda x: "warm" if x else "cold")
    df["solve_strategy"] = df[["solver", "temp"]].apply(lambda x: "/".join(x.astype(str)), axis=1)

    select_datasets = np.array(sorted([s for s in df["dataset name"].unique().tolist()
                                       if int(re.sub("[a-z,A-Z]+", "", s)) > 121
                                       and s not in ["pr124", "si175", "brg180",
                                                     "bier127", "tho150", "ch130", "u159"]]))

    subset_df = df[df["dataset name"].isin(select_datasets.tolist())]
    subset_df = subset_df.sort_values(by="dataset name", kind="quicksort")
    subset_df = subset_df.sort_values(by="solve_strategy", kind="quicksort")

    palette = sns.color_palette()
    ax = sns.barplot(
        data=subset_df,
        x="dataset name",
        y="best relative gap",
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

    plt.xlabel("dataset")
    plt.yscale("log")
    sns.despine()
    ax.set_ylim((0, 16))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
          ncol=4, fancybox=True, shadow=True)
    ax.tick_params(width=2)
    plt.setp(ax.spines.values(), linewidth=2)
    plt.tight_layout()
    plt.savefig("qap_bar_chart.png")
    plt.clf()
    os.system("convert qap_bar_chart.png -trim qap_bar_chart.png")