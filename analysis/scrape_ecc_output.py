from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import pandas as pd
import re
import seaborn as sns

from IPython import embed


def create_df(log_fname: str) -> pd.DataFrame:
    start_time = None
    solve_time = None
    skip_flag = False

    solver_name = None
    solve_times = defaultdict(list)

    with open(log_fname, "r") as f:
        for line in f:
            line.strip()

            re_new_block = re.search(">> loaded block \"(.*)\"", line)
            if re_new_block:
                print("@@@ current block: ", re_new_block.group(1))
                start_time = None
                skip_flag = True

            re_round_0_complete = re.search("Round 0", line)
            if re_round_0_complete:
                skip_flag = False

            if not skip_flag:
                re_solver_name = re.search(">>>>> START:\s+(.+)$", line)
                if re_solver_name:
                    solver_name = re_solver_name.group(1)

                if start_time is None:
                    re_start_time = re.search("start_time:\s([\.0-9]+)", line)
                    if re_start_time:
                        start_time = float(re_start_time.group(1))

                re_end_time = re.search("end_time:\s([\.0-9]+)", line)
                if re_end_time:
                    solve_time = float(re_end_time.group(1)) - start_time

                re_solver_name = re.search("<<<<< END:\s+(.+)$", line)
                if re_solver_name:
                    assert solver_name == re_solver_name.group(1)
                    solve_times[solver_name].append(solve_time)
                    print("{}: {}".format(solver_name, solve_time))
                    start_time = None

    dataset_rename = {"pubmed": "Pubmed", "qian": "QIAN", "zbmath": "SCAD-zbMATH"}

    df = pd.DataFrame(
        columns=(
            "dataset",
            "solver",
            "warm-start",
            "num ecc",
            "per-iteration time",
            "cumulative time",
            "speedup"
        )
    )
    i = 0
    for name, vals in solve_times.items():
        cum_time = 0.0
        for j, val in enumerate(vals):
            cum_time += val
            df.loc[i] = [
                dataset_rename[os.path.basename(log_fname).split(".")[0]],
                "CGAL" if name.split("/")[0] == "cgal" else "SpecBM",
                "True" if name.split("/")[1] == "warm" else "False",
                j+1,
                val,
                cum_time,
                solve_times[name.replace("warm", "cold")][j] / val if name.split("/")[1] == "warm" else 1.0
            ]
            i += 1

    return df


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 16})

    #pubmed_df = create_df(log_fname="results/ecc/pubmed.out")
    #qian_df = create_df(log_fname="results/ecc/qian.out")
    #zbmath_df = create_df(log_fname="results/ecc/zbmath.out")

    pubmed_df = create_df(log_fname="pubmed.out")
    qian_df = create_df(log_fname="qian.out")
    zbmath_df = create_df(log_fname="zbmath.out")

    ax = sns.lineplot(
        pubmed_df,
        x="num ecc",
        y="cumulative time",
        hue="solver",
        style="warm-start",
        linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("# of $\exists$-constraints")
    plt.ylabel("cumulative SDP solve time (sec)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("pubmed.png")
    plt.clf()
    os.system("convert pubmed.png -trim pubmed.png")

    ax = sns.lineplot(
        qian_df,
        x="num ecc",
        y="cumulative time",
        hue="solver",
        style="warm-start",
        linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.get_legend().set_visible(False)
    plt.xlabel("# of $\exists$-constraints")
    plt.ylabel("cumulative SDP solve time (sec)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("qian.png")
    plt.clf()
    os.system("convert qian.png -trim qian.png")

    ax = sns.lineplot(
        zbmath_df,
        x="num ecc",
        y="cumulative time",
        hue="solver",
        style="warm-start",
        linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.get_legend().set_visible(False)
    plt.xlabel("# of $\exists$-constraints")
    plt.ylabel("cumulative SDP solve time (sec)")
    plt.grid()
    plt.tight_layout()
    plt.savefig("zbmath.png")
    plt.clf()
    os.system("convert zbmath.png -trim zbmath.png")

    df = pd.concat([pubmed_df, qian_df, zbmath_df]).reset_index(drop=True)
    warm_start_df = df[df["warm-start"] == "True"]

    g = sns.catplot(
        data=warm_start_df,
        x="solver",
        y="speedup",
        col="dataset",
        kind="bar",
        aspect=0.5,
        capsize=0.1,
        errwidth=3.0)
    g.set_axis_labels("", "warm-start solve time \nfold change")
    g.set_titles("{col_name}")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("warm_start_time_reduction.png")
    plt.clf()
    os.system("convert warm_start_time_reduction.png -trim warm_start_time_reduction.png")

    embed()
    exit()