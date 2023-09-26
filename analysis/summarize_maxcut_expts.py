import ast
from collections import defaultdict
import glob
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

                re_iter_info = re.search("t:\s(\d+).*end_time:\s([\.0-9]+).*obj_gap:\s([\.0-9]+)"
                                         ".*infeas_gap:\s([\.0-9]+).*max_infeas:\s([\.0-9]+)"
                                         ".*callback_val:\s([\.0-9]+)", line)
                if re_iter_info:
                    iteration.append(int(re_iter_info.group(1)))
                    time.append(float(re_iter_info.group(2)) - start_time)
                    objective_gap.append(float(re_iter_info.group(3)))
                    infeasibility_gap.append(float(re_iter_info.group(4)))
                    max_infeasibility.append(float(re_iter_info.group(5)))
                    callback_val.append(float(re_iter_info.group(6)))
    
    df = pd.DataFrame(
            columns=(
                *hparam_names,
                "time (sec)",
                "iteration",
                "objective residual",
                "infeasibility gap",
                "max infeasibility",
                "callback value"
            )
    )

    if len(iteration) < 1:
        embed()
        exit()

    time = [t - time[0] + 0.1 for t in time]
    df["time (sec)"] = time
    df["iteration"] = iteration
    df["objective residual"] = iteration
    df["infeasibility gap"] = infeasibility_gap
    df["max infeasibility"] = max_infeasibility
    df["callback value"] = callback_val

    for hparam_key in hparam_names:
        df[hparam_key] = hparam_dict[hparam_key]

    return df


if __name__ == "__main__":
    
    expt_name = "maxcut_G67_B"
    expt_out_files = glob.glob(f"results/*{expt_name}*")

    dfs = []
    for fname in tqdm(expt_out_files):
        dfs.append(create_df_from_log(fname))

    embed()
    exit()

    merged_df = pd.concat(dfs)
    
    # dump aggregated df to pickle file
    with open(f"analysis/sims_summary_df_{expt_name}.pkl", "wb") as f:
        pickle.dump(merged_df, f)

    embed()
    exit()