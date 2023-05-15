from collections import defaultdict
import glob
import os
import pickle
import re

import numpy as np
import pandas as pd

from IPython import embed


def create_df_from_log(log_fname):

    begin = False
    warm_start = ""
    solver = ""
    start_time = None
    time = []
    iteration = []
    # TODO: maybe change this to be best so far?
    feasibility_gap = []
    objective_gap = []
    cut_value = []

    with open(log_fname, "r") as f:
        for line in f:
            line.strip()

            if warm_start == "":
                re_warm_start = re.search("WARM_START:\s+([A-Za-z]+)", line)
                if re_warm_start:
                    warm_start = re_warm_start.group(1)

            if solver == "":
                re_solver = re.search("SOLVER:\s+([A-Za-z]+)", line)
                if re_solver:
                    solver = re_solver.group(1)

            if not begin:
                re_begin = re.search("\sBEGIN\s", line)
                if re_begin:
                    begin = True
            else:
                if start_time is None:
                    re_start_time = re.search("start_time:\s([\.0-9]+)", line)
                    if re_start_time:
                        start_time = float(re_start_time.group(1))

                re_iter_info = re.search("t:\s(\d+).*end_time:\s([\.0-9]+).*obj_gap:\s([\.0-9]+)"
                                         ".*infeas_gap:\s([\.0-9]+).*callback_val:\s([\.0-9]+)", line)
                if re_iter_info:
                    iteration.append(int(re_iter_info.group(1)))
                    time.append(float(re_iter_info.group(2)) - start_time)
                    objective_gap.append(float(re_iter_info.group(3)))
                    feasibility_gap.append(float(re_iter_info.group(4)))
                    cut_value.append(float(re_iter_info.group(5)))

    df = pd.DataFrame(
            columns=(
                "solver",       # solver name with or without warm-start
                "time (sec)",
                "iteration",
                "objective residual",
                "feasibility",
                "weight of cut"
            )
    )

    print("Filling dataframe for file: {} ...".format(log_fname))
    for i in range(1, len(time)):
        df.loc[i] = [
            solver + " (warm-start)" if warm_start == "True" else solver + " (no warm-start)",
            time[i],
            iteration[i],
            objective_gap[i],
            feasibility_gap[i],
            cut_value[i]
        ]
    print("Done.")

    return df

if __name__ == "__main__":
    
    expt_out_files = [
        "../results/cgal_no_warm_start_G67.out",
        "../results/cgal_warm_start_0_99_G67.out",
        "../results/specbm_no_warm_start_G67.out",
        "../results/specbm_warm_start_0_99_G67.out"
    ]

    dfs = []
    for fname in expt_out_files:
        dfs.append(create_df_from_log(fname))

    merged_df = pd.concat(dfs)
    
    # dump aggregated df to pickle file
    with open('sims_summary_df.pkl', 'wb') as f:
        pickle.dump(merged_df, f)

    embed()
    exit()
    