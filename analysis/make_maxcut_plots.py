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

    tmp_df = df[df["warm-start"] == True][["warm_start_strategy", "warm_start_frac"]].drop_duplicates()
    warm_start_tuples = list(tmp_df.itertuples(index=False, name=None))

    all_data_paths = df["data_path"].unique()
    for data_path in all_data_paths:
        for k_curr in [k for k in df["k_curr"].unique() if k is not None]:
            for k_past in [k for k in df["k_past"].unique() if k is not None]:
                for warm_start_strategy, warm_start_frac in warm_start_tuples:
                    dataset_basename = os.path.basename(data_path).split(".")[0]
                    cgal_mask = (df["solver"] == "cgal")
                    k_curr_mask = (df["k_curr"] == k_curr)
                    k_past_mask = (df["k_past"] == k_past)
                    warm_start_strategy_mask = (df["warm_start_strategy"] == warm_start_strategy)
                    warm_start_strategy_mask |= (df["warm_start_strategy"] == "none")
                    warm_start_frac_mask = (df["warm_start_frac"] == warm_start_frac)
                    warm_start_frac_mask |= (df["warm_start_frac"] == 1.0)
                    data_path_mask = (df["data_path"] == data_path)

                    subset_mask = cgal_mask | (k_curr_mask & k_past_mask)
                    subset_mask &= data_path_mask
                    subset_mask &= (warm_start_strategy_mask & warm_start_frac_mask)
                    subset_df = df[subset_mask]

                    ax = sns.lineplot(
                        subset_df,
                        x="time (sec)",
                        y="smooth objective residual",
                        hue="solver",
                        hue_order=["cgal", "specbm"],
                        style="warm-start",
                        style_order=[False, True],
                        linewidth=1)

                    warm_start_str = warm_start_strategy + "-" + str(warm_start_frac).replace(".", "_")

                    plt.xscale("log")
                    plt.yscale("log")
                    plt.grid()
                    #plt.show()
                    imgname = "time-vs-obj-{}-{}-{}-{}.png".format(
                        dataset_basename, k_curr, k_past, warm_start_str)
                    print(f"Saving plot to {imgname}...")
                    plt.savefig(imgname)
                    plt.clf()
                    #os.system("convert pubmed.png -trim pubmed.png")

                    ax = sns.lineplot(
                        subset_df,
                        x="time (sec)",
                        y="smooth infeasibility gap",
                        hue="solver",
                        hue_order=["cgal", "specbm"],
                        style="warm-start",
                        style_order=[False, True],
                        linewidth=1)

                    plt.xscale("log")
                    plt.yscale("log")
                    plt.grid()
                    #plt.show()
                    imgname = "time-vs-infeas-{}-{}-{}-{}.png".format(
                        dataset_basename, k_curr, k_past, warm_start_str)
                    print(f"Saving plot to {imgname}...")
                    plt.savefig(imgname)
                    plt.clf()
                    #os.system("convert pubmed.png -trim pubmed.png")

                    ax = sns.lineplot(
                        subset_df,
                        x="time (sec)",
                        y="callback value",
                        hue="solver",
                        hue_order=["cgal", "specbm"],
                        style="warm-start",
                        style_order=[False, True],
                        linewidth=1)

                    plt.xscale("log")
                    plt.grid()
                    #plt.show()
                    imgname = "time-vs-cut-{}-{}-{}-{}.png".format(
                        dataset_basename, k_curr, k_past, warm_start_str)
                    print(f"Saving plot to {imgname}...")
                    plt.savefig(imgname)
                    plt.clf()
                    #os.system("convert pubmed.png -trim pubmed.png")