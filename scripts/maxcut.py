import argparse
import cvxpy as cp
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
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
    parser.add_argument('--data_path', type=str, required=True, help="path to mat file")
    parser.add_argument('--warm_start', action='store_true', help="warm-start or not")
    parser.add_argument('--warm_start_frac', type=float,
                        help="fraction of data used to warmkstart")
    parser.add_argument('--solver', type=str, required=True, choices=["specbm", "cgal"],
                        help="name of solver to use")
    parser.add_argument('--warm_start_max_iters', type=int,
                        help="number of iterations to run warm-start")
    parser.add_argument('--max_iters', type=int, required=True,
                        help="number of iterations to run solver")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":

    hparams = get_hparams()

    # variables controlling experiment
    MAT_PATH = hparams.data_path
    WARM_START = hparams.warm_start
    WARM_START_FRAC = hparams.warm_start_frac
    SOLVER = hparams.solver
    K = 10                       # number of eigenvectors to compute for specbm
    R = 100                     # size of the sketch
    LANCZOS_MAX_RESTARTS = 10
    EPS = 1e-5
    WARM_START_MAX_ITERS = hparams.warm_start_max_iters
    MAX_ITERS = hparams.max_iters

    # print out all of the variable for this experiment
    print("MAT_PATH: ", MAT_PATH)
    print("WARM_START_FRAC: ", WARM_START_FRAC)
    print("WARM_START: ", WARM_START)
    print("SOLVER: ", SOLVER)
    print("K: ", K)
    print("R: ", R)
    print("LANCZOS_MAX_RESTARTS: ", LANCZOS_MAX_RESTARTS)
    print("EPS: ", EPS)
    print("WARM_START_MAX_ITERS: ", WARM_START_MAX_ITERS)
    print("MAX_ITERS: ", MAX_ITERS)

    jax.config.update("jax_enable_x64", True)

    # load the problem data
    try:
        problem = loadmat(MAT_PATH)
        dict_format = False
    except:
        problem = mat73_loadmat(MAT_PATH)
        dict_format = True

    if "Gset" in MAT_PATH:
        C = problem["Problem"][0][0][1]
    elif "DIMACS" in MAT_PATH and not dict_format:
        C = problem["Problem"][0][0][2]
    elif "DIMACS" in MAT_PATH and dict_format:
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
    Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, R))

    if Omega is None:
        X = jnp.zeros((n, n))
        P = None
    else:
        X = None
        P = jnp.zeros((n, R))

    y = jnp.zeros((n,))
    z = jnp.zeros((n,))
    tr_X = 0.0
    primal_obj = 0.0

    k_curr = K
    k_past = 1
    k = k_curr + k_past

    print("\n+++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++\n")

    if SOLVER == "cgal":
        SCALE_X = 1.0 / float(n)
        SCALE_C = 1.0 / scipy.sparse.linalg.norm(C, ord="fro") 
    elif SOLVER == "specbm":
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
    C = C.tocoo()
    C = BCOO(
        (C.data, jnp.stack((C.row, C.col)).T), shape=C.shape)

    trace_ub = 1.0 * float(n) * SCALE_X
    m = n

    A_data = jnp.ones((n))
    A_indices = jnp.stack(
        (jnp.arange(n), jnp.arange(n), jnp.arange(n))).T 
    b = jnp.ones((n,)) * SCALE_X

    if SOLVER == "specbm":
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
            k_curr=k_curr,
            k_past=k_past,
            SCALE_C=1.0,
            SCALE_X=1.0,
            eps=EPS,
            max_iters=MAX_ITERS,
            lanczos_inner_iterations=min(n, max(2*k + 1, 32)),
            lanczos_max_restarts=LANCZOS_MAX_RESTARTS,
            subprob_tol=1e-7,
            callback_fn=compute_max_cut)
    elif SOLVER == "cgal":
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
            C_matvec=C_matvec,
            A_operator_slim=A_operator_slim,
            A_adjoint_slim=A_adjoint_slim,
            Omega=Omega,
            b=b,
            beta0=1.0,
            SCALE_C=SCALE_C,
            SCALE_X=SCALE_X,
            eps=EPS,
            max_iters=MAX_ITERS,
            lanczos_inner_iterations=32,
            lanczos_max_restarts=10,
            subprob_tol=1e-10,
            callback_fn=compute_max_cut)
    else:
        raise ValueError("Invalid SOLVER")