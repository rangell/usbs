from collections import defaultdict
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import embed


if __name__ == "__main__":
    log_fname = "results/ecc/qian.out"

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

    df = pd.DataFrame(
        columns=(
            "solver",
            "warm-start",
            "num ecc",
            "per-iteration time",
            "cumulative time"
        )
    )
    i = 0
    for name, vals in solve_times.items():
        cum_time = 0.0
        for j, val in enumerate(vals):
            cum_time += val
            df.loc[i] = [
                "CGAL" if name.split("/")[0] == "cgal" else "SpecBM",
                "True" if name.split("/")[1] == "warm" else "False",
                j+1,
                val,
                cum_time
            ]
            i += 1


    sns.lineplot(df, x="num ecc", y="cumulative time", hue="solver", style="warm-start")
    #plt.title("zbmath")
    plt.xlabel("# of $\exists$-constraints")
    plt.ylabel("cumulative SDP solve time (s)")
    plt.grid()
    plt.show()

    #plt.savefig("zbmath.png")