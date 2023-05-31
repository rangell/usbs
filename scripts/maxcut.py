import argparse
import cvxpy as cp
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import json
import numpy as np
from pathlib import Path
import pickle
import scipy  # type: ignore
from scipy.io import loadmat  # type: ignore
from mat73 import loadmat as mat73_loadmat
from scipy.sparse import coo_matrix, csc_matrix  # type: ignore
from typing import Any, Callable, Tuple

from solver.cgal import cgal
from solver.specbm import specbm
from solver.utils import reconstruct_from_sketch

from IPython import embed


@jax.jit
def compute_max_cut(C: BCOO, Omega: Array, P: Array) -> int:
    W, _ = reconstruct_from_sketch(Omega, P)
    W_bin = 2 * (W > 0).astype(float) - 1
    return jnp.max(jnp.diag(-W_bin.T @ C @ W_bin))


def solve_scs(C: csc_matrix) -> np.ndarray[Any, Any]:
    n = C.shape[0]
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == np.ones((n,))]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)
    X_scs = X.value
    return X_scs


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--data_path", type=str, required=True, help="path to mat file")
    parser.add_argument("--solver", type=str, required=True, choices=["specbm", "cgal"],
                        help="name of solver to use")
    parser.add_argument("--max_iters", type=int, required=True,
                        help="number of iterations to run solver")
    parser.add_argument("--k_curr", type=int, required=True,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--k_past", type=int, required=True,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--r", type=int, required=True, help="sketch dimension")
    parser.add_argument("--lanczos_max_restarts", type=int, required=True,
                        help="max number of restarts for thick restart Lanczos")
    parser.add_argument("--eps", type=float, required=True,
                        help="error tolerance for solver convergence")
    parser.add_argument("--subprob_tol", type=float, default=1e-7,
                        help="error tolerance for Lanczos and IPMs")
    parser.add_argument("--warm_start", action='store_true', help="warm-start or not")
    parser.add_argument("--warm_start_frac", type=float,
                        help="fraction of data used to warm-start")
    parser.add_argument("--warm_start_max_iters", type=int,
                        help="number of iterations to run warm-start")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":

    # get experiment hparams and print them out
    hparams = get_hparams()
    print(json.dumps(vars(hparams), indent=4))

    jax.config.update("jax_enable_x64", True)

    # load the problem data
    try:
        problem = loadmat(hparams.data_path)
        dict_format = False
    except:
        problem = mat73_loadmat(hparams.data_path)
        dict_format = True

    if "Gset" in hparams.data_path:
        C = problem["Problem"][0][0][1]
    elif "DIMACS" in hparams.data_path and not dict_format:
        C = problem["Problem"][0][0][2]
    elif "DIMACS" in hparams.data_path and dict_format:
        C = problem["Problem"]["A"]
    else:
        raise ValueError("Unknown path type")

    n = C.shape[0]

    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = -0.25*C
    C = C.tocsc()

    ## solve with SCS if we have not already
    #scs_soln_cache = str(Path(MAT_PATH).with_suffix("")) + "_scs_soln.pkl"
    #if Path(scs_soln_cache).is_file():
    #    with open(scs_soln_cache, "rb") as f_in:
    #        X_scs = pickle.load(f_in)
    #else:
    #    X_scs = solve_scs(C)
    #    with open(scs_soln_cache, "wb") as f_out:
    #        pickle.dump(X_scs, f_out)
        
    # construct the test matrix for the sketch
    Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, hparams.r))

    if Omega is None:
        X = jnp.zeros((n, n))
        P = None
    else:
        X = None
        P = jnp.zeros((n, hparams.r))

    y = jnp.zeros((n,))
    z = jnp.zeros((n,))
    tr_X = 0.0
    primal_obj = 0.0

    k = hparams.k_curr + hparams.k_past

     #### Do the warm-start
    if hparams.warm_start:
        print("\n+++++++++++++++++++++++++++++ WARM-START ++++++++++++++++++++++++++++++++++\n")

        warm_start_n = int(hparams.warm_start_frac * n)
        warm_start_C = C.tolil()[:warm_start_n, :warm_start_n].tocsr()

        if hparams.solver == "cgal":
            WARM_START_SCALE_X = 1.0 / float(warm_start_n)
            WARM_START_SCALE_C = 1.0 / scipy.sparse.linalg.norm(warm_start_C, ord="fro") 
        elif hparams.solver == "specbm":
            WARM_START_SCALE_X = 1.0
            WARM_START_SCALE_C = 1.0

        warm_start_m = warm_start_n
        warm_start_trace_ub = 1.0 * float(warm_start_n) * WARM_START_SCALE_X

        scaled_warm_start_C = warm_start_C.tocoo().T * WARM_START_SCALE_C
        scaled_warm_start_C = BCOO(
            (scaled_warm_start_C.data,
            jnp.stack((scaled_warm_start_C.row, scaled_warm_start_C.col)).T),
            shape=scaled_warm_start_C.shape)

        warm_start_A_data = jnp.ones((warm_start_n))
        warm_start_A_indices = jnp.stack(
            (jnp.arange(warm_start_n), jnp.arange(warm_start_n), jnp.arange(warm_start_n))).T 
        warm_start_b = jnp.ones((warm_start_n,)) * WARM_START_SCALE_X

        if Omega is None:
            warm_start_X = jnp.zeros((warm_start_n, warm_start_n))
            warm_start_Omega = None
            warm_start_P = None
        else:
            warm_start_X = None
            warm_start_Omega = Omega[:warm_start_n, :]
            warm_start_P = jnp.zeros((warm_start_n, hparams.r))
        warm_start_y = jnp.zeros((warm_start_m,))
        warm_start_z = jnp.zeros((warm_start_n,))

        if hparams.solver == "specbm":
            (warm_start_X,
             warm_start_P,
             warm_start_y,
             warm_start_z,
             warm_start_primal_obj,
             warm_start_tr_X) = specbm(
                X=warm_start_X,
                P=warm_start_P,
                y=warm_start_y,
                z=warm_start_z,
                primal_obj=0.0,
                tr_X=0.0,
                n=warm_start_n,
                m=warm_start_m,
                trace_ub=warm_start_trace_ub,
                C=scaled_warm_start_C,
                A_data=warm_start_A_data,
                A_indices=warm_start_A_indices,
                b=warm_start_b,
                Omega=warm_start_Omega,
                rho=0.5,
                beta=0.25,
                k_curr=hparams.k_curr,
                k_past=hparams.k_past,
                SCALE_C=1.0,
                SCALE_X=1.0,
                eps=hparams.eps,
                max_iters=hparams.warm_start_max_iters,
                lanczos_inner_iterations=min(n, max(2*k + 1, 32)),
                lanczos_max_restarts=hparams.lanczos_max_restarts,
                subprob_tol=1e-7,
                callback_fn=compute_max_cut)
        elif hparams.solver == "cgal":
            (warm_start_X,
             warm_start_P,
             warm_start_y,
             warm_start_z,
             warm_start_primal_obj,
             warm_start_tr_X) = cgal(
                X=warm_start_X,
                P=warm_start_P,
                y=warm_start_y,
                z=warm_start_z,
                primal_obj=0.0,
                tr_X=0.0,
                n=warm_start_n,
                m=warm_start_m,
                trace_ub=warm_start_trace_ub,
                C=scaled_warm_start_C,
                A_data=warm_start_A_data,
                A_indices=warm_start_A_indices,
                b=warm_start_b,
                Omega=Omega,
                beta0=1.0,
                SCALE_C=WARM_START_SCALE_C,
                SCALE_X=WARM_START_SCALE_X,
                eps=hparams.eps,
                max_iters=hparams.warm_start_max_iters,
                lanczos_inner_iterations=min(n, max(2*k + 1, 32)),
                lanczos_max_restarts=hparams.lanczos_max_restarts,
                subprob_tol=1e-7,
                callback_fn=compute_max_cut)
        else:
            raise ValueError("Invalid SOLVER")

        if Omega is None:
            X = X.at[:warm_start_n, :warm_start_n].set(warm_start_X)
        else:
            P = P.at[:warm_start_n, :].set(warm_start_P)
        y = y.at[:warm_start_n].set(warm_start_y)
        z = z.at[:warm_start_n].set(warm_start_z)
        tr_X = warm_start_tr_X
        primal_obj = warm_start_primal_obj / (WARM_START_SCALE_C * WARM_START_SCALE_X)

    print("\n+++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++\n")

    if hparams.solver == "cgal":
        SCALE_X = 1.0 / float(n)
        SCALE_C = 1.0 / scipy.sparse.linalg.norm(C, ord="fro") 
    elif hparams.solver == "specbm":
        SCALE_X = 1.0
        SCALE_C = 1.0
    else:
        raise ValueError("Invalid SOLVER")

    # rescale the primal objective (for cgal)
    primal_obj *= (SCALE_X * SCALE_C)

    scaled_C = C * SCALE_C
    scaled_C = scaled_C.tocoo().T
    scaled_C = BCOO(
        (scaled_C.data, jnp.stack((scaled_C.row, scaled_C.col)).T), shape=scaled_C.shape)

    trace_ub = 1.0 * float(n) * SCALE_X
    m = n

    A_data = jnp.ones((n))
    A_indices = jnp.stack(
        (jnp.arange(n), jnp.arange(n), jnp.arange(n))).T 
    b = jnp.ones((n,)) * SCALE_X

    if hparams.solver == "specbm":
        X, P, y, z, primal_obj, tr_X = specbm(
            X=X,
            P=P,
            y=y,
            z=z,
            primal_obj=primal_obj,
            tr_X=tr_X,
            n=n,
            m=n,
            trace_ub=trace_ub,
            C=scaled_C,
            A_data=A_data,
            A_indices=A_indices,
            b=b,
            Omega=Omega,
            rho=0.5,
            beta=0.25,
            k_curr=hparams.k_curr,
            k_past=hparams.k_past,
            SCALE_C=1.0,
            SCALE_X=1.0,
            eps=hparams.eps,
            max_iters=hparams.max_iters,
            lanczos_inner_iterations=min(n, max(2*k + 1, 32)),
            lanczos_max_restarts=hparams.lanczos_max_restarts,
            subprob_tol=1e-7,
            callback_fn=compute_max_cut)
    elif hparams.solver == "cgal":
        X, P, y, z, primal_obj, tr_X = cgal(
            X=X,
            P=P,
            y=y,
            z=z,
            primal_obj=primal_obj,
            tr_X=tr_X,
            n=n,
            m=m,
            trace_ub=trace_ub,
            C=scaled_C,
            A_data=A_data,
            A_indices=A_indices,
            b=b,
            Omega=Omega,
            beta0=1.0,
            SCALE_C=SCALE_C,
            SCALE_X=SCALE_X,
            eps=hparams.eps,
            max_iters=hparams.max_iters,
            lanczos_inner_iterations=min(n, max(2*k + 1, 32)),
            lanczos_max_restarts=hparams.lanczos_max_restarts,
            subprob_tol=1e-7,
            callback_fn=compute_max_cut)
    else:
        raise ValueError("Invalid SOLVER")