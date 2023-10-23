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
    plt.rcParams.update({"font.size": 10, "figure.figsize": (30, 5)})
    plt.gcf().subplots_adjust(left=0.3, bottom=0.3)

    hparams = get_hparams()

    with open(hparams.summary_df_fname, "rb") as f:
        df = pickle.load(f)

    df["dataset name"] = [os.path.basename(x).split(".")[0] for x in df["data_path"]]
    df["warm-start"] = (df["warm_start_strategy"] != "none")
    df["temp"] = df["warm-start"].apply(lambda x: "warm" if x else "cold")
    df["solve_strategy"] = df[["solver", "temp"]].apply(lambda x: "/".join(x.astype(str)), axis=1)

    # TODO: which datasets should we show in bar chart

    select_datasets = np.array(sorted([s for s in df["dataset name"].unique().tolist()
                                       if int(re.sub("[a-z,A-Z]+", "", s)) > 135
                                       and s not in ["pr124", "kroA150", "tai150b", "brg180", "tho150"]])[::-1])

    subset_df = df[df["dataset name"].isin(select_datasets.tolist())]
    subset_df = subset_df.sort_values(by="solve_strategy")
    subset_df = subset_df.sort_values(by="dataset name")

    palette = sns.color_palette()
    ax = sns.barplot(
        data=subset_df,
        x="dataset name",
        y="best relative gap",
        hue="solve_strategy",
        palette=[palette[0], palette[0], palette[1], palette[1]],
        #palette=palette,
        errorbar=None,
        edgecolor='black')

    for bars, hatch, legend_handle in zip(ax.containers, ['', '//', '', '//'], ax.legend_.legend_handles):
        for bar, color in zip(bars, palette):
            bar.set_hatch(hatch)
        # update the existing legend, use twice the hatching pattern to make it denser
        legend_handle.set_hatch(hatch + hatch)

    plt.yscale("log")
    sns.despine()
    plt.show()

    embed()
    exit()