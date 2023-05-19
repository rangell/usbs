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
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix  # type: ignore
from typing import Any, Callable, Tuple

from solver.specbm_slow import specbm_slow

from IPython import embed


def solve_scs(C: csc_matrix) -> np.ndarray[Any, Any]:
    n = C.shape[0]
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == np.ones((n,))]
    constraints += [X >= 0]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)
    X_scs = X.value
    return X_scs


if __name__ == "__main__":

    np.random.seed(0)

    MAT_PATH = "data/maxcut/DIMACS10/chesapeake.mat"
    WARM_START = False
    WARM_START_FRAC = 1.0
    SOLVER = "specbm"
    K_CURR = 5                       
    K_PAST = 2
    R = 100                     # size of the sketch
    LANCZOS_NUM_ITERS = 300     
    EPS = 1e-7
    WARM_START_MAX_ITERS = 100
    MAX_ITERS = 100

    # print out all of the variable for this experiment
    print("MAT_PATH: ", MAT_PATH)
    print("WARM_START_FRAC: ", WARM_START_FRAC)
    print("WARM_START: ", WARM_START)
    print("SOLVER: ", SOLVER)
    print("K_CURR: ", K_CURR)
    print("K_PAST: ", K_PAST)
    print("R: ", R)
    print("LANCZOS_NUM_ITERS: ", LANCZOS_NUM_ITERS)
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

    n_orig = C.shape[0]

    edge_signs = 2 * np.random.binomial(1, 0.25, C.nnz) - 1
    edge_weights = edge_signs + np.random.normal(scale=0.5, size=(C.nnz,))
    C.data *= edge_weights
    C = -0.5*(C + C.T)      # make matrix symmetric and negate edges for minimization
    n = int(n_orig + (C.nnz / 2))  # NOTE: this assumes the diagonals are all zero 

    #X_scs = solve_scs(C)

    C = C.tocoo()
    C = BCOO((C.data, jnp.stack((C.row, C.col)).T), shape=(n, n))

    trace_ub = 1.0 * float(n)

    X = jnp.zeros((n, n))
    y = jnp.zeros((n,))
    z = jnp.zeros((n,))

    tr_X = 0.0
    primal_obj = 0.0

    k_curr = K_CURR
    k_past = K_PAST
    k = k_curr + k_past

    # create constraint linear operator
    A_data = jnp.ones((n_orig))
    A_indices = jnp.stack(
        (jnp.arange(n_orig), jnp.arange(n_orig), jnp.arange(n_orig))).T 
    A_data_new = []
    A_indices_new = []
    i = 0
    for row_col_idx in C.indices:
        r = int(row_col_idx[0])
        c = int(row_col_idx[1])
        if r < c:
            A_data_new.append(0.5)
            A_indices_new.append([i+n_orig, r, c])
            A_data_new.append(0.5)
            A_indices_new.append([i+n_orig, c, r])
            A_data_new.append(-1.0)
            A_indices_new.append([i+n_orig, i+n_orig, i+n_orig])
            i += 1

    A_data = jnp.concatenate([A_data, jnp.array(A_data_new)], axis=0)
    A_indices = jnp.concatenate([A_indices, jnp.array(A_indices_new)], axis=0)

    A_tensor = BCOO((A_data, A_indices), shape=(n, n, n)).todense()

    def apply_A_operator_slim(u: Array):
        outvec = jnp.zeros((n,))
        outvec = outvec.at[A_indices[:,0]].add(
            A_data * u.at[A_indices[:,1]].get() * u.at[A_indices[:,2]].get())
        return outvec

    def apply_A_adjoint_slim(z: Array, u: Array) -> Array:
        outvec = jnp.zeros((n,))
        outvec = outvec.at[A_indices[:,2]].add(
            A_data * z.at[A_indices[:,0]].get() * u.at[A_indices[:,1]].get())
        return outvec
    
    v1 = jax.random.normal(jax.random.PRNGKey(1), (n,))
    v2 = jax.random.normal(jax.random.PRNGKey(2), (n,))
    v3 = jax.random.normal(jax.random.PRNGKey(3), (n,))

    M1 = v1.reshape((-1, 1)) @ v1.reshape((1, -1))
    M2 = v2.reshape((-1, 1)) @ v2.reshape((1, -1))
    M12 = M1 + M2

    z1_slow = jnp.sum(jnp.sum(A_tensor * M1.reshape((1, n, n)), axis=2), axis=1)
    z1_fast = apply_A_operator_slim(v1)

    # TODO: fix this! we don't want constant folding with A_data & A_indices!
    A_operator_batched = jax.vmap(apply_A_operator_slim, 1, 1)

    embed()
    exit()

    b = jnp.concatenate([jnp.ones((n_orig,)), jnp.zeros((n - n_orig,))])
    m = b.size

    # Testing a different initialization
    # IMPORTANT: `jnp.diag(X)` is not necessarily `z`!
    X = X.at[(jnp.arange(n), jnp.arange(n))].set(-b)
    z = -b
    tr_X = jnp.trace(X)

    # make everything dense for ease of use with SCS
    C = C.todense()
    A_tensor = A_tensor.todense()

    X, y, z, primal_obj, tr_X = specbm_slow(
        X=X,
        y=y,
        z=z,
        primal_obj=primal_obj,
        tr_X=tr_X,
        n=n,
        m=m,
        trace_ub=trace_ub,
        C=C,
        A_tensor=A_tensor,
        b=b,
        rho=0.5,
        beta=0.25,
        k_curr=k_curr,
        k_past=k_past,
        SCALE_C=1.0,
        SCALE_X=1.0,
        eps=EPS,
        max_iters=MAX_ITERS,
        lanczos_num_iters=LANCZOS_NUM_ITERS,
        callback_fn=None)