import argparse
import git
import jax
import jax.numpy as jnp
import json
from mat73 import loadmat as mat73_loadmat
import pickle
from scipy.io import loadmat  # type: ignore
import sys

from solver.specbm import specbm
from utils.maxcut_helpers import (initialize_state,
                                  compute_max_cut,
                                  get_implicit_warm_start_state,
                                  get_explicit_warm_start_state,
                                  get_dual_only_warm_start_state)

from IPython import embed


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--data_path", type=str, required=True, help="path to mat file")
    parser.add_argument("--max_iters", type=int, required=True,
                        help="number of iterations to run solver")
    parser.add_argument("--max_time", type=float, default=jnp.inf,
                        help="max running time in seconds for one solve")
    parser.add_argument("--obj_gap_eps", type=float, default=-jnp.inf,
                        help="early stop if obj_gap is less than this number")
    parser.add_argument("--infeas_gap_eps", type=float, default=-jnp.inf,
                        help="early stop if infeas_gap is less than this number")
    parser.add_argument("--max_infeas_eps", type=float, default=-jnp.inf,
                        help="early stop if max_infeas is less than this number")
    parser.add_argument("--lanczos_max_restarts", type=int, default=100,
                        help="number of restarts to use for thick restart lanczos")
    parser.add_argument("--lanczos_max_inner_iterations", type=int, default=32,
                        help="number of inner iterations to use for thick restart lanczos")
    parser.add_argument("--subprob_eps", type=float, default=1e-7,
                        help="error tolerance for IPM, alternating minimization, and lanczos")
    parser.add_argument("--subprob_max_iters", type=int, default=15,
                        help="max iters for IPM and alternating minimization")
    parser.add_argument("--k_curr", type=int, default=1,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--k_past", type=int, default=0,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--trace_factor", type=float, default=1.0,
                        help="how much space to give trace")
    parser.add_argument("--rho", type=float, default=0.1,
                        help="proximal parameter")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="sufficient decrease parameter")
    parser.add_argument("--warm_start_frac", type=float, default=1.0,
                        help="fraction of dimensions to drop for warm-starting")
    parser.add_argument("--sketch_dim", type=int, default=0,
                        help="dimension of Nystrom sketch")
    parser.add_argument("--warm_start_strategy", type=str,
                        choices=["implicit", "explicit", "dual_only", "none"],
                        help="warm-start strategy to use")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # get experiment hparams and print them out
    hparams = get_hparams()
    hparams.solver = "specbm"
    print("cmd: ", " ".join(["python"] + sys.argv))
    print("git sha: ", git.Repo(search_parent_directories=True).head.object.hexsha)
    print("hparams: ", json.dumps(vars(hparams), indent=4))

    assert hparams.warm_start_strategy == "none" or hparams.warm_start_frac < 1.0

    DATAFILE = hparams.data_path
    try:
        problem = loadmat(DATAFILE)
        dict_format = False
    except:
        problem = mat73_loadmat(DATAFILE)
        dict_format = True
    if "Gset" in DATAFILE:
        C = problem["Problem"][0][0][1]
    elif "DIMACS" in DATAFILE and not dict_format:
        C = problem["Problem"][0][0][2]
    elif "DIMACS" in DATAFILE and dict_format:
        C = problem["Problem"]["A"]
    else:
        raise ValueError("Unknown path type")

    hparams.num_drop = C.shape[0] - int(C.shape[0] * hparams.warm_start_frac)
    n = C.shape[0] - hparams.num_drop
    C = C.tolil()[:n, :n].tocsc()

    # set sketch_dim = -1 if we do not want sketch
    sdp_state = initialize_state(C=C, sketch_dim=hparams.sketch_dim)

    trace_ub = hparams.trace_factor * float(n) * sdp_state.SCALE_X

    k_curr = hparams.k_curr
    k_past = hparams.k_past

    if hparams.num_drop == 0:
        print("\n+++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++\n")
    else:
        print("\n+++++++++++++++++++++++++++++ WARM-START ++++++++++++++++++++++++++++++++++\n")

    sdp_state = specbm(
        sdp_state=sdp_state,
        n=sdp_state.C.shape[0],
        m=sdp_state.b.shape[0],
        trace_ub=trace_ub,
        trace_factor=hparams.trace_factor,
        rho=hparams.rho,
        beta=hparams.beta,
        k_curr=k_curr,
        k_past=k_past,
        max_iters=hparams.max_iters,
        max_time=hparams.max_time,
        obj_gap_eps=hparams.obj_gap_eps,
        infeas_gap_eps=hparams.infeas_gap_eps,
        max_infeas_eps=hparams.max_infeas_eps,
        lanczos_inner_iterations=min(sdp_state.C.shape[0], hparams.lanczos_max_inner_iterations),
        lanczos_max_restarts=hparams.lanczos_max_restarts,
        subprob_eps=hparams.subprob_eps,
        subprob_max_iters=hparams.subprob_max_iters,
        callback_fn=compute_max_cut,
        callback_static_args=pickle.dumps(None),
        callback_nonstatic_args=sdp_state.C / sdp_state.SCALE_C)

    if hparams.num_drop == 0:
        exit()

    DATAFILE = hparams.data_path
    try:
        problem = loadmat(DATAFILE)
        dict_format = False
    except:
        problem = mat73_loadmat(DATAFILE)
        dict_format = True
    if "Gset" in DATAFILE:
        C = problem["Problem"][0][0][1]
    elif "DIMACS" in DATAFILE and not dict_format:
        C = problem["Problem"][0][0][2]
    elif "DIMACS" in DATAFILE and dict_format:
        C = problem["Problem"]["A"]
    else:
        raise ValueError("Unknown path type")

    if hparams.warm_start_strategy == "implicit":
        sdp_state = get_implicit_warm_start_state(
            old_sdp_state=sdp_state, C=C, sketch_dim=hparams.sketch_dim)
    elif hparams.warm_start_strategy == "explicit":
        sdp_state = get_explicit_warm_start_state(
            old_sdp_state=sdp_state, C=C, sketch_dim=hparams.sketch_dim)
    elif hparams.warm_start_strategy == "dual_only":
        sdp_state = get_dual_only_warm_start_state(
            old_sdp_state=sdp_state, C=C, sketch_dim=hparams.sketch_dim)
    else:
        raise NotImplementedError("Warm-start strategy not implemented.")

    trace_ub = hparams.trace_factor * float(sdp_state.C.shape[0]) * sdp_state.SCALE_X

    k_curr = hparams.k_curr
    k_past = hparams.k_past

    print("\n+++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++\n")

    sdp_state = specbm(
        sdp_state=sdp_state,
        n=sdp_state.C.shape[0],
        m=sdp_state.b.shape[0],
        trace_ub=trace_ub,
        trace_factor=hparams.trace_factor,
        rho=hparams.rho,
        beta=hparams.beta,
        k_curr=k_curr,
        k_past=k_past,
        max_iters=hparams.max_iters,
        max_time=hparams.max_time,
        obj_gap_eps=hparams.obj_gap_eps,
        infeas_gap_eps=hparams.infeas_gap_eps,
        max_infeas_eps=hparams.max_infeas_eps,
        lanczos_inner_iterations=min(sdp_state.C.shape[0], hparams.lanczos_max_inner_iterations),
        lanczos_max_restarts=hparams.lanczos_max_restarts,
        subprob_eps=hparams.subprob_eps,
        subprob_max_iters=hparams.subprob_max_iters,
        callback_fn=compute_max_cut,
        callback_static_args=pickle.dumps(None),
        callback_nonstatic_args=sdp_state.C / sdp_state.SCALE_C)