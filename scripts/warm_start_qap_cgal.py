import argparse
import jax
import json
import pickle

from solver.cgal import cgal
from utils.qap_helpers import (load_and_process_qap,
                               load_and_process_tsp,
                               qap_round,
                               initialize_state,
                               get_implicit_warm_start_state,
                               get_explicit_warm_start_state,
                               get_dual_only_warm_start_state)

from IPython import embed


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--data_path", type=str, required=True, help="path to mat file")
    parser.add_argument("--max_iters", type=int, required=True,
                        help="number of iterations to run solver")
    parser.add_argument("--trace_factor", type=float, default=1.0,
                        help="how much space to give trace")
    parser.add_argument("--num_drop", type=int, default=0,
                        help="number of dimensions to drop for warm-starting")
    parser.add_argument("--warm_start_strategy", type=str,
                        choices=["implicit", "explicit", "dual_only", "none"],
                        help="warm-start strategy to use")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # get experiment hparams and print them out
    hparams = get_hparams()
    print(json.dumps(vars(hparams), indent=4))

    assert hparams.warm_start_strategy == "none" or hparams.num_drop > 0

    DATAFILE = hparams.data_path
    if DATAFILE.split(".")[-1] == "dat":
        l, D, W, C = load_and_process_qap(DATAFILE, num_drop=hparams.num_drop)
    elif DATAFILE.split(".")[-1] == "tsp":
        l, D, W, C = load_and_process_tsp(DATAFILE, num_drop=hparams.num_drop)
    else:
        raise ValueError("Invalid data file type.")

    # set sketch_dim = -1 if we do not want sketch
    sdp_state = initialize_state(C=C, sketch_dim=l+hparams.num_drop)

    trace_ub = hparams.trace_factor * float(l + 1) * sdp_state.SCALE_X

    callback_static_args = pickle.dumps({"l": l})
    callback_nonstatic_args = {"D": D, "W": W}

    if hparams.num_drop == 0:
        print("\n+++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++\n")
    else:
        print("\n+++++++++++++++++++++++++++++ WARM-START ++++++++++++++++++++++++++++++++++\n")

    sdp_state = cgal(
        sdp_state=sdp_state,
        n=sdp_state.C.shape[0],
        m=sdp_state.b.shape[0],
        trace_ub=trace_ub,
        beta0=1.0,
        eps=1e-5,  # hparams.eps,
        max_iters=hparams.max_iters,
        lanczos_inner_iterations=min(sdp_state.C.shape[0], 32),
        lanczos_max_restarts=100,  # hparams.lanczos_max_restarts,
        subprob_eps=1e-7,
        callback_fn=qap_round,
        callback_static_args=callback_static_args,
        callback_nonstatic_args=callback_nonstatic_args)

    if hparams.num_drop == 0:
        exit()

    DATAFILE = hparams.data_path
    if DATAFILE.split(".")[-1] == "dat":
        l, D, W, C = load_and_process_qap(DATAFILE, num_drop=0)
    elif DATAFILE.split(".")[-1] == "tsp":
        l, D, W, C = load_and_process_tsp(DATAFILE, num_drop=0)
    else:
        raise ValueError("Invalid data file type.")

    if hparams.warm_start_strategy == "implicit":
        sdp_state = get_implicit_warm_start_state(old_sdp_state=sdp_state, C=C, sketch_dim=l)
    elif hparams.warm_start_strategy == "explicit":
        sdp_state = get_explicit_warm_start_state(old_sdp_state=sdp_state, C=C, sketch_dim=l)
    elif hparams.warm_start_strategy == "dual_only":
        sdp_state = get_dual_only_warm_start_state(old_sdp_state=sdp_state, C=C, sketch_dim=l)
    else:
        raise NotImplementedError("Warm-start strategy not implemented.")

    trace_ub = hparams.trace_factor * float(l + 1) * sdp_state.SCALE_X

    callback_static_args = pickle.dumps({"l": l})
    callback_nonstatic_args = {"D": D, "W": W}

    print("\n+++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++\n")

    sdp_state = cgal(
        sdp_state=sdp_state,
        n=sdp_state.C.shape[0],
        m=sdp_state.b.shape[0],
        trace_ub=trace_ub,
        beta0=1.0,
        eps=1e-5,  # hparams.eps,
        max_iters=hparams.max_iters,
        lanczos_inner_iterations=min(sdp_state.C.shape[0], 32),
        lanczos_max_restarts=100,  # hparams.lanczos_max_restarts,
        subprob_eps=1e-7,
        callback_fn=qap_round,
        callback_static_args=callback_static_args,
        callback_nonstatic_args=callback_nonstatic_args)