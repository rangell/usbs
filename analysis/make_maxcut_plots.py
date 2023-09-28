import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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
    hparams = get_hparams()

    with open(hparams.summary_df_fname, "rb") as f:
        df = pickle.load(f)

    df["warm-start"] = (df["warm_start_strategy"] != "none")

    plt.rcParams.update({'font.size': 16})

    ax = sns.lineplot(
        df,
        x="time (sec)",
        y="objective residual",
        hue="solver",
        style="warm-start",
        linewidth=3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("time (sec)")
    plt.ylabel("objective residual")
    #plt.xscale("log")
    plt.yscale("log")
    plt.grid()
    plt.show()
    #plt.savefig("time-vs-obj.png")
    plt.clf()
    #os.system("convert pubmed.png -trim pubmed.png")

    embed()
    exit()