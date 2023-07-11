import argparse
import jax
import json
import pickle

from solver.specbm import specbm
from utils.qap_helpers import (load_and_process_qap,
                               load_and_process_tsp,
                               qap_round,
                               initialize_state,
                               get_implicit_warm_start_state)

from IPython import embed


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--data_path", type=str, required=True, help="path to mat file")
    parser.add_argument("--max_iters", type=int, required=True,
                        help="number of iterations to run solver")
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
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # get experiment hparams and print them out
    hparams = get_hparams()
    print(json.dumps(vars(hparams), indent=4))

    #DATAFILE = "data/qap/qapdata/chr12a.dat"
    #DATAFILE = "data/qap/tspdata/ulysses16.tsp"
    #DATAFILE = "data/qap/tspdata/dantzig42.tsp"
    #DATAFILE = "data/qap/tspdata/bayg29.tsp"
    #DATAFILE = "data/qap/tspdata/bays29.tsp"
    #DATAFILE = "data/qap/tspdata/att48.tsp"

    # for warm-start, TODO: turn this into an input parameter
    num_drop = 1

    DATAFILE = hparams.data_path
    if DATAFILE.split(".")[-1] == "dat":
        l, D, W, C = load_and_process_qap(DATAFILE, num_drop=num_drop)
    elif DATAFILE.split(".")[-1] == "tsp":
        l, D, W, C = load_and_process_tsp(DATAFILE, num_drop=num_drop)
    else:
        raise ValueError("Invalid data file type.")

    # set sketch_dim = -1 if we do not want sketch
    sdp_state = initialize_state(C=C, sketch_dim=l+num_drop)

    trace_ub = hparams.trace_factor * float(l + 1) * sdp_state.SCALE_X

    k_curr = hparams.k_curr
    k_past = hparams.k_past

    callback_static_args = pickle.dumps({"l": l})
    callback_nonstatic_args = {"D": D, "W": W}

    sdp_state = specbm(
        sdp_state=sdp_state,
        n=sdp_state.C.shape[0],
        m=sdp_state.b.shape[0],
        trace_ub=trace_ub,
        rho=hparams.rho,
        beta=hparams.beta,
        k_curr=k_curr,
        k_past=k_past,
        eps=1e-5,  # hparams.eps,
        max_iters=hparams.max_iters,  # hparams.max_iters,
        lanczos_inner_iterations=min(sdp_state.C.shape[0], 32),
        lanczos_max_restarts=100,  # hparams.lanczos_max_restarts,
        subprob_eps=1e-7,
        subprob_max_iters=15,
        callback_fn=qap_round,
        callback_static_args=callback_static_args,
        callback_nonstatic_args=callback_nonstatic_args)

    DATAFILE = hparams.data_path
    if DATAFILE.split(".")[-1] == "dat":
        l, D, W, C = load_and_process_qap(DATAFILE, num_drop=0)
    elif DATAFILE.split(".")[-1] == "tsp":
        l, D, W, C = load_and_process_tsp(DATAFILE, num_drop=0)
    else:
        raise ValueError("Invalid data file type.")

    sdp_state = get_implicit_warm_start_state(old_sdp_state=sdp_state, C=C, sketch_dim=l)

    trace_ub = hparams.trace_factor * float(l + 1) * sdp_state.SCALE_X

    k_curr = hparams.k_curr
    k_past = hparams.k_past

    callback_static_args = pickle.dumps({"l": l})
    callback_nonstatic_args = {"D": D, "W": W}

    sdp_state = specbm(
        sdp_state=sdp_state,
        n=sdp_state.C.shape[0],
        m=sdp_state.b.shape[0],
        trace_ub=trace_ub,
        rho=hparams.rho,
        beta=hparams.beta,
        k_curr=k_curr,
        k_past=k_past,
        eps=1e-5,  # hparams.eps,
        max_iters=hparams.max_iters,  # hparams.max_iters,
        lanczos_inner_iterations=min(sdp_state.C.shape[0], 32),
        lanczos_max_restarts=100,  # hparams.lanczos_max_restarts,
        subprob_eps=1e-7,
        subprob_max_iters=15,
        callback_fn=qap_round,
        callback_static_args=callback_static_args,
        callback_nonstatic_args=callback_nonstatic_args)