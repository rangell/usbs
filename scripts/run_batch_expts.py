import itertools
import pickle
from types import SimpleNamespace

from IPython import embed


if __name__ == "__main__":

    expt_config = {
        "expt_name": "test_sweep",
        "problem": "maxcut",
        "hparam_grid": {
            "solver": ["cgal", "specbm"],
            "data_path": ["data/maxcut/Gset/G1.mat"],
            "max_iters": [10000],
            "max_time": [360],
            "obj_gap_eps": [1e-7],
            "infeas_gap_eps": [1e-7],
            "max_infeas_eps": [1e-7],
            "trace_factor": [2.0],
            "warm_start_frac": [0.9],
            "k_curr": [3],
            "k_past": [1],
            "rho": [0.5],
            "beta": [0.25],
            "sketch_dim": [10],
            "subprob_max_iters": [100],
            "subprob_eps": [1e-15],
            "lanczos_max_restarts": [100],
            "warm_start_strategy": ["none", "implicit"]
        }
    }
    cgal_exclude = ["solver", "k_curr", "k_past", "rho", "beta"]

    keys, values = zip(*expt_config["hparam_grid"].items())
    for d in [dict(zip(keys, v)) for v in itertools.product(*values)]:

        cmd_str = "python scripts/warm_start_{}_{}.py ".format(expt_config["problem"], d["solver"])
        if d["solver"] == "cgal":
            cmd_str += " ".join([f"--{k}={v}" for k, v in d.items() if k not in cgal_exclude])
        elif d["solver"] == "specbm":
            cmd_str += " ".join([f"--{k}={v}" for k, v in d.items() if k not in ["solver"]])

        print(f"cmd: {cmd_str}\n")
