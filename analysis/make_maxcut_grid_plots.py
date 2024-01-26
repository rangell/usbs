import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pickle
import seaborn as sns
import os

from IPython import embed


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--summary_df_fname", type=str, required=True, help="summary dataframe filename")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})
    plt.gcf().subplots_adjust(left=0.2, bottom=0.15)

    hparams = get_hparams()

    with open(hparams.summary_df_fname, "rb") as f:
        df = pickle.load(f)

    assert len(df["data_path"].unique()) == 1
    dataset_basename = os.path.basename(df["data_path"].unique()[0]).split(".")[0]
    for k_past in [k for k in df["k_past"].unique() if k is not None]:
        k_past_mask = (df["k_past"] == k_past)
        cold_start_mask = (df["warm_start_strategy"] == "none")
        subset_mask = k_past_mask & cold_start_mask
        subset_df = df[subset_mask]

        ax = sns.lineplot(
            subset_df,
            x="time (sec)",
            y="objective residual",
            hue="k_curr",
            hue_order=sorted(subset_df["k_curr"].unique()),
            palette="colorblind",
            linewidth=2)

        plt.legend(title="$k_c$", loc="lower left")
        plt.ylabel("")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()

        if k_past != 0:
            ax.get_legend().set_visible(False)

        print("k_past: ", k_past)
        imgname = "time-vs-obj-gpu-grid-{}-k_p-{}.png".format(dataset_basename, k_past)
        print(f"Saving plot to {imgname}...")
        plt.savefig(imgname)
        plt.clf()
        os.system(f"convert {imgname} -trim {imgname}")

        ax = sns.lineplot(
            subset_df,
            x="time (sec)",
            y="infeasibility gap",
            hue="k_curr",
            hue_order=sorted(subset_df["k_curr"].unique()),
            palette="colorblind",
            linewidth=2)

        plt.legend(title="$k_c$", loc="lower left")
        plt.ylabel("")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid()

        if k_past != 0:
            ax.get_legend().set_visible(False)

        print("k_past: ", k_past)
        imgname = "time-vs-infeas-gpu-grid-{}-k_p-{}.png".format(dataset_basename, k_past)
        print(f"Saving plot to {imgname}...")
        plt.savefig(imgname)
        plt.clf()
        os.system(f"convert {imgname} -trim {imgname}")