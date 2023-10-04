import argparse
import ast
from collections import defaultdict
import glob
import json
import os
import pandas as pd
import pickle
import re
from tqdm import tqdm, trange

from IPython import embed


def create_df_from_log(log_fname):

    begin = False
    # hparam_names are different for different types of data
    hparam_names = [
        "data_path",
        "max_iters",
        "max_time",
        "obj_gap_eps",
        "infeas_gap_eps",
        "max_infeas_eps",
        "lanczos_max_restarts",
        "subprob_eps",
        "subprob_max_iters",
        "k_curr",
        "k_past",
        "trace_factor",
        "rho",
        "beta",
        "num_drop",
        "warm_start_strategy",
        "solver"]
    hparam_dict = defaultdict(lambda : None)
    start_time = None
    time = []
    iteration = []
    infeasibility_gap = []
    max_infeasibility = []
    objective_gap = []
    callback_val = []
    min_callback_val = None
    best_callback_val = []

    with open(log_fname, "r") as f:
        for line in f:
            line.strip()

            for hparam_key in hparam_names:
                if hparam_key not in hparam_dict.keys():
                    re_hparam_key = re.search(f"\"{hparam_key}\":\s+([^,]+).*$", line)
                    if re_hparam_key:
                        try:
                            hparam_dict[hparam_key] = ast.literal_eval(re_hparam_key.group(1).strip())
                        except:
                            hparam_dict[hparam_key] = re_hparam_key.group(1).strip()

            if not begin:
                re_begin = re.search("\sBEGIN\s", line)
                if re_begin:
                    begin = True
            else:
                if start_time is None:
                    re_start_time = re.search("start_time:\s([\.0-9]+)", line)
                    if re_start_time:
                        start_time = float(re_start_time.group(1))

                re_iter_info = re.search("t:\s(\d+).*end_time:\s([\.0-9]+).*obj_gap:\s([-e\.0-9]+)"
                                         ".*infeas_gap:\s([-e\.0-9]+).*max_infeas:\s([-e\.0-9]+)"
                                         ".*callback_val:\s([-e\.0-9]+)", line)
                if re_iter_info:
                    iteration.append(int(re_iter_info.group(1)))
                    time.append(float(re_iter_info.group(2)) - start_time)
                    objective_gap.append(float(re_iter_info.group(3)))
                    infeasibility_gap.append(float(re_iter_info.group(4)))
                    max_infeasibility.append(float(re_iter_info.group(5)))
                    callback_val.append(float(re_iter_info.group(6)))
                    if  min_callback_val is None or float(re_iter_info.group(6)) < min_callback_val:
                        if float(re_iter_info.group(6)) > 0:
                            min_callback_val = float(re_iter_info.group(6))
                    best_callback_val.append(min_callback_val)

    
    df = pd.DataFrame(
        columns=(
            *hparam_names,
            "time (sec)",
            "iteration",
            "objective residual",
            "infeasibility gap",
            "max infeasibility",
            "callback value",
            "best callback value",
            "relative gap",
            "best relative gap"
        )
    )

    if len(iteration) < 1:
        embed()
        exit()

    time = [t - time[0] + 0.1 for t in time]
    df["time (sec)"] = time
    df["iteration"] = iteration
    df["objective residual"] = objective_gap
    df["infeasibility gap"] = infeasibility_gap
    df["max infeasibility"] = max_infeasibility
    df["callback value"] = callback_val
    df["best callback value"] = best_callback_val

    for hparam_key in hparam_names:
        df[hparam_key] = hparam_dict[hparam_key]

    data_path = df["data_path"].unique()
    assert len(data_path) == 1
    data_path = data_path[0]

    if "qapdata" in data_path:
        soln_path = data_path.split(".")[0] + ".sln"
        with open(soln_path, "r") as f:
            opt_soln = int(f.read().strip().split()[1])
        df["relative gap"] = (df["callback value"] - opt_soln) / opt_soln
        df["best relative gap"] = (df["best callback value"] - opt_soln) / opt_soln
    elif "tspdata" in data_path:
        with open("data/qap/tspdata/tsp_solns.json", "r") as f:
            tsp_solns = json.load(f)
        opt_soln = tsp_solns[os.path.basename(data_path).split(".")[0]]
        df["relative gap"] = (df["callback value"] - opt_soln) / opt_soln
        df["best relative gap"] = (df["best callback value"] - opt_soln) / opt_soln
    else:
        raise ValueError("Data path does not belong to a known set")
    
    if hparam_dict["warm_start_strategy"] == "none":
        return df.iloc[1:]
    else:
        return df


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--expt_name", type=str, required=True, help="experiment basename")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    
    hparams = get_hparams()
    expt_out_files = glob.glob(f"results/qap/{hparams.expt_name}*.out")

    dfs = []
    for fname in tqdm(expt_out_files):
        dfs.append(create_df_from_log(fname))

    merged_df = pd.concat(dfs).reset_index(drop=True)
    
    summary_df_fname = f"results/qap/{hparams.expt_name}.pkl"
    print(f"Writing summary df: {summary_df_fname}...")
    with open(summary_df_fname, "wb") as f:
        pickle.dump(merged_df, f)
    print("Done.")