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

    for k_curr in [k for k in df["k_curr"].unique() if k is not None]:
        for k_past in [k for k in df["k_past"].unique() if k is not None]:
            for warm_start_strategy in ["implicit", "explicit", "dual_only"]:
                cgal_mask = (df["solver"] == "cgal")
                k_curr_mask = (df["k_curr"] == k_curr)
                k_past_mask = (df["k_past"] == k_past)
                warm_start_strategy_mask = (df["warm_start_strategy"] == warm_start_strategy)
                warm_start_strategy_mask |= (df["warm_start_strategy"] == "none")

                subset_mask = cgal_mask | (k_curr_mask & k_past_mask)
                subset_mask &= warm_start_strategy_mask
                subset_df = df[subset_mask]

                ax = sns.lineplot(
                    subset_df,
                    x="time (sec)",
                    y="objective residual",
                    hue="solver",
                    style="warm-start",
                    linewidth=1)

                plt.xscale("log")
                plt.yscale("log")
                plt.grid()
                #plt.show()
                imgname = "time-vs-obj-{}-{}-{}.png".format(k_curr, k_past, warm_start_strategy)
                print(f"Saving plot to {imgname}...")
                plt.savefig(imgname)
                plt.clf()
                #os.system("convert pubmed.png -trim pubmed.png")

                ax = sns.lineplot(
                    subset_df,
                    x="time (sec)",
                    y="infeasibility gap",
                    hue="solver",
                    style="warm-start",
                    linewidth=1)

                plt.xscale("log")
                plt.yscale("log")
                plt.grid()
                #plt.show()
                imgname = "time-vs-infeas-{}-{}-{}.png".format(k_curr, k_past, warm_start_strategy)
                print(f"Saving plot to {imgname}...")
                plt.savefig(imgname)
                plt.clf()
                #os.system("convert pubmed.png -trim pubmed.png")

                ax = sns.lineplot(
                    subset_df,
                    x="time (sec)",
                    y="callback value",
                    hue="solver",
                    style="warm-start",
                    linewidth=1)

                plt.xscale("log")
                plt.grid()
                #plt.show()
                imgname = "time-vs-cut-{}-{}-{}.png".format(k_curr, k_past, warm_start_strategy)
                print(f"Saving plot to {imgname}...")
                plt.savefig(imgname)
                plt.clf()
                #os.system("convert pubmed.png -trim pubmed.png")