import argparse
import itertools
import json
import os
import tempfile

from IPython import embed


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--config", type=str, required=True, help="path to config file")
    parser.add_argument("--sbatch_template", type=str, required=True, help="path to sbatch template")
    parser.add_argument("--safe_mode", action="store_true", help="runs script without submitting jobs")
    hparams = parser.parse_args()
    return hparams

if __name__ == "__main__":
    hparams = get_hparams()

    with open(hparams.config, "r") as f:
        expt_config = json.load(f)

    with open(hparams.sbatch_template, "r") as f:
        sbatch_template = f.read()

    cgal_exclude = ["solver", "k_curr", "k_past", "rho", "beta"]

    submitted_cmds = set()

    keys, values = zip(*expt_config["hparam_grid"].items())
    for d in [dict(zip(keys, v)) for v in itertools.product(*values)]:
        if expt_config["problem"] == "maxcut":
            d["warm_start_frac"], d["warm_start_strategy"] = d["warm_start"]
            del d["warm_start"]
        elif expt_config["problem"] == "qap":
            d["num_drop"], d["warm_start_strategy"] = d["warm_start"]
            del d["warm_start"]
        else:
            raise ValueError("Unsupported problem type")

        cmd_str = "python scripts/warm_start_{}_{}.py ".format(expt_config["problem"], d["solver"])
        if d["solver"] == "cgal":
            cmd_str += " ".join([f"--{k}={v}" for k, v in d.items() if k not in cgal_exclude])
        elif d["solver"] == "specbm":
            cmd_str += " ".join([f"--{k}={v}" for k, v in d.items() if k not in ["solver"]])

        if cmd_str in submitted_cmds:
            continue

        submitted_cmds.add(cmd_str)
        job_name = "{}.{}".format(expt_config["expt_name"], str(len(submitted_cmds)))
        out_path = "results/{}/{}".format(expt_config["problem"], job_name)

        sbatch_str = sbatch_template.replace("__job_name__", job_name)
        sbatch_str = sbatch_str.replace("__out_path__", out_path)
        sbatch_str = sbatch_str.replace("__cmd_str__", cmd_str)
        
        print(f"cmd: {cmd_str}\n")
        with tempfile.NamedTemporaryFile() as f:
            f.write(bytes(sbatch_str.strip(), "utf-8"))
            f.seek(0)
            if not hparams.safe_mode:
                os.system(f"sbatch {f.name}")