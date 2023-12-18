import argparse
import ast
from collections import defaultdict
import glob
import math
import pandas as pd
import pickle
import re
from scipy.ndimage import gaussian_filter1d
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
        "warm_start_frac",
        "sketch_dim",
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

    with open(log_fname, "r") as f:
        for line in f:
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
                                         ".*callback_val:\s(.+)\s", line)
                if re_iter_info:
                    iteration.append(int(re_iter_info.group(1)))
                    time.append(float(re_iter_info.group(2)) - start_time)
                    objective_gap.append(float(re_iter_info.group(3)))
                    infeasibility_gap.append(float(re_iter_info.group(4)))
                    max_infeasibility.append(float(re_iter_info.group(5)))
                    if "None" in re_iter_info.group(6):
                        callback_val.append(float(-1.0))
                    else:
                        callback_val.append(float(re_iter_info.group(6)))
    
    df = pd.DataFrame(
        columns=(
            *hparam_names,
            "time (sec)",
            "iteration",
            "objective residual",
            "infeasibility gap",
            "max infeasibility",
            "callback value",
        )
    )

    if len(iteration) == 0:
        print(f"Failed: {log_fname}, dataset: {hparam_dict['data_path']}")
        return None, hparam_dict["data_path"]

    log_indices = [j*(2**i) for j in range(3) for i in range(int(math.log2(len(iteration))) + 1)]
    log_indices = set(log_indices)
    log_indices = sorted([i for i in log_indices if i < len(iteration)])
    log_indices.append(len(iteration) - 1)

    time = [t - time[0] + 1.0 for t in time]
    df["time (sec)"] = [time[i] for i in log_indices]
    df["iteration"] = [iteration[i] for i in log_indices]
    df["objective residual"] = [objective_gap[i] for i in log_indices]
    df["infeasibility gap"] = [infeasibility_gap[i] for i in log_indices]
    df["max infeasibility"] = [max_infeasibility[i] for i in log_indices]
    df["callback value"] = [callback_val[i] for i in log_indices]

    for hparam_key in hparam_names:
        if hparam_key == "solver":
            df[hparam_key] = "CGAL" if hparam_dict[hparam_key] == "cgal" else "SpecBM"
        else:
            df[hparam_key] = hparam_dict[hparam_key]

    return df, hparam_dict["data_path"]


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--expt_name", type=str, required=True, help="experiment basename")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    
    hparams = get_hparams()
    expt_out_files = glob.glob(f"results/maxcut/{hparams.expt_name}*.out")

    df_tuples = []
    for fname in tqdm(expt_out_files):
        print(f"fname: {fname}")
        df_tuples.append(create_df_from_log(fname))

    dropped_data_paths = set([x[1] for x in df_tuples if x[0] is None])
    dfs = [x[0] for x in df_tuples if x[1] not in dropped_data_paths]

    merged_df = pd.concat(dfs).reset_index(drop=True)
    slim_merged_df = pd.concat([df.iloc[-1:] for df in dfs]).reset_index(drop=True)

    summary_df_fname = f"results/maxcut/{hparams.expt_name}.pkl"
    print(f"Writing summary df: {summary_df_fname}...")
    with open(summary_df_fname, "wb") as f:
        pickle.dump(merged_df, f)
    print("Done.")

    slim_summary_df_fname = f"results/maxcut/{hparams.expt_name}.slim.pkl"
    print(f"Writing slim summary df: {slim_summary_df_fname}...")
    with open(slim_summary_df_fname, "wb") as f:
        pickle.dump(slim_merged_df, f)
    print("Done.")