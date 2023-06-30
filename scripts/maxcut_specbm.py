import jax
from jax import lax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from mat73 import loadmat as mat73_loadmat
import numpy as np
import scipy  # type: ignore
from scipy.io import loadmat  # type: ignore
from scipy.spatial.distance import pdist, squareform  # type: ignore
from typing import Any, Callable, Tuple

from scripts.munkres import munkres
from solver.specbm import specbm
from solver.utils import reconstruct_from_sketch

from IPython import embed


@jax.jit
def compute_max_cut(C: BCOO, Omega: Array, P: Array) -> int:
    W, _ = reconstruct_from_sketch(Omega, P)
    W_bin = 2 * (W > 0).astype(float) - 1
    return jnp.max(jnp.diag(-W_bin.T @ C @ W_bin))


def get_all_problem_data(C: BCOO) -> Tuple[BCOO, Array, Array, Array]:
    n = C.shape[0]
    range_n = jnp.arange(n)[:, None]
    A_indices = jnp.hstack(3*[range_n])
    A_data = jnp.ones((n,))
    b = jnp.ones((n,))
    b_ineq_mask = jnp.zeros((n,))
    return A_indices, A_data, b, b_ineq_mask


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    DATAFILE = "data/maxcut/Gset/G1.mat"
    R = 100

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

    # convert to Laplacian
    n = C.shape[0]
    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = -0.25*C
    C = BCOO.from_scipy_sparse(C)

    # get constraint specification
    A_indices, A_data, b, b_ineq_mask = get_all_problem_data(C)
    m = b.shape[0]

    SCALE_X = 1.0 / float(n)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = jnp.zeros((m,))
    SCALE_A = SCALE_A.at[A_indices[:,0]].add(A_data**2)
    SCALE_A = 1.0 / jnp.sqrt(SCALE_A)

    scaled_C = BCOO((C.data * SCALE_C, C.indices), shape=C.shape)
    scaled_b = b * SCALE_X * SCALE_A
    scaled_A_data = A_data * SCALE_A.at[A_indices[:,0]].get()

    #X = jnp.zeros((n, n))
    #Omega = None
    #P = None
    X = None
    Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, R))
    P = jnp.zeros_like(Omega)
    y = jnp.zeros((m,))
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    trace_ub = 2.0 * float(n) * SCALE_X

    k_curr = 5
    k_past = 2

    X, P, y, z, primal_obj, tr_X = specbm(
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
        A_data=scaled_A_data,
        A_indices=A_indices,
        b=scaled_b,
        b_ineq_mask=b_ineq_mask,
        Omega=Omega,
        rho=0.5,
        beta=0.25,
        k_curr=k_curr,
        k_past=k_past,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A,
        eps=1e-3,  # hparams.eps,
        max_iters=10000,  # hparams.max_iters,
        lanczos_inner_iterations=min(n, 32),
        lanczos_max_restarts=100,  # hparams.lanczos_max_restarts,
        subprob_eps=1e-7,
        subprob_max_iters=15,
        callback_fn=compute_max_cut)

    embed()
    exit()