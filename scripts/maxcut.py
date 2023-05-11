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
from scipy.sparse import coo_matrix, csc_matrix  # type: ignore
from typing import Any, Callable

from solver.cgal import cgal
from solver.specbm import specbm
from solver.eigen import approx_grad_k_min_eigen

from IPython import embed


def create_C_innerprod(C: BCOO) -> Callable[[Array], float]:
    @jax.jit
    def C_innerprod(X: Array) -> float:
        return jnp.trace(C @ X)
    return C_innerprod


def create_C_add(C: BCOO) -> Callable[[Array], Array]:
    dense_C = C.todense()
    @jax.jit
    def C_add(X: Array) -> Array:
        return dense_C + X
    return C_add


def create_C_matvec(C: BCOO) -> Callable[[Array], Array]:
    @jax.jit
    def C_matvec(u: Array) -> Array:
        return C @ u
    return C_matvec


def create_A_operator() -> Callable[[Array], Array]:
    @jax.jit
    def A_operator(X: Array) -> Array:
        return jnp.diag(X)
    return A_operator


def create_A_operator_slim() -> Callable[[Array, Array], Array]:
    @jax.jit
    def A_operator_slim(u: Array) -> Array:
        return u ** 2
    return A_operator_slim


def create_A_adjoint(n: int) -> Callable[[Array], Array]:
    @jax.jit
    def A_adjoint(z: Array) -> Array:
        Y = jnp.zeros((n,n))
        Y = Y.at[jnp.diag_indices(n, ndim=2)].set(z)
        return Y
    return A_adjoint


def create_A_adjoint_slim() -> Callable[[Array, Array], Array]:
    @jax.jit
    def A_adjoint(z: Array, u: Array) -> Array:
        return z * u
    return A_adjoint


def create_proj_K(n: int, SCALE_X: float) -> Callable[[Array], Array]:
    @jax.jit
    def proj_K(z: Array) -> Array:
        return jnp.ones((n,)) * SCALE_X
    return proj_K

# TODO: put this in some utils file
def create_svec_matrix(k: int) -> BCOO:
    U = np.zeros((int(0.5*k*(k+1)), k**2))
    for a, (b, c) in enumerate(list(zip(*np.tril_indices(k)))):
        if b == c:
            U[a, b*k + c] = 1.0
        else:
            U[a, b*k + c] = 1.0 / np.sqrt(2.0)
            U[a, c*k + b] = U[a, b*k + c]
    U = coo_matrix(U)
    U = BCOO((U.data, jnp.stack((U.row, U.col)).T), shape=U.shape)
    return U


def create_Q_base(m: int, k: int, U: BCOO) -> Callable[[Array], Array]:
    @jax.jit
    def Q_base(V: Array) -> Array:
        flat_outer_prod = (V.T.reshape(1, k, m) * V.T.reshape(k, 1, m)).reshape(k**2, m)
        svec_proj = U @ flat_outer_prod
        expanded_mx = svec_proj.reshape(-1, 1, m) * svec_proj.reshape(1, -1, m)
        final_mx = jnp.sum(expanded_mx, axis=-1)
        return final_mx
    return Q_base


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


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    np.random.seed(0)
    MAT_PATH = "./data/maxcut/Gset/G1.mat"
    problem = loadmat(MAT_PATH)
    C = problem["Problem"][0][0][1]
    n = C.shape[0]
    m = n

    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = -0.25*C
    C = C.tocsc()

    #SCALE_C = 1.0 / scipy.sparse.linalg.norm(C, ord="fro") 
    #SCALE_X = 1.0 / n
    #trace_ub = 1.0
    SCALE_C = 1.0
    SCALE_X = 1.0
    trace_ub = 1.0*float(n)

    scs_soln_cache = str(Path(MAT_PATH).with_suffix("")) + "_scs_soln.pkl"
    if Path(scs_soln_cache).is_file():
        with open(scs_soln_cache, "rb") as f_in:
            X_scs = pickle.load(f_in)
    else:
        X_scs = solve_scs(C)
        with open(scs_soln_cache, "wb") as f_out:
            pickle.dump(X_scs, f_out)

    scaled_C = C * SCALE_C
    scaled_C = scaled_C.tocoo().T
    scaled_C = BCOO(
        (scaled_C.data, jnp.stack((scaled_C.row, scaled_C.col)).T), shape=scaled_C.shape)

    C_innerprod = create_C_innerprod(scaled_C)
    C_add = create_C_add(scaled_C)
    C_matvec = create_C_matvec(scaled_C)
    A_operator = create_A_operator()
    A_operator_slim = create_A_operator_slim()
    A_adjoint = create_A_adjoint(n)
    A_adjoint_slim = create_A_adjoint_slim()
    proj_K = create_proj_K(n, SCALE_X)
    b = jnp.ones((n,)) * SCALE_X

    #X, y = cgal(
    #   n=n,
    #   m=n,
    #   trace_ub=trace_ub,
    #   C_matvec=C_matvec,
    #   A_operator_slim=A_operator_slim,
    #   A_adjoint_slim=A_adjoint_slim,
    #   proj_K=proj_K,
    #   beta0=1.0,
    #   SCALE_C=SCALE_C,
    #   SCALE_X=SCALE_X,
    #   eps=1e-3,
    #   max_iters=10000,
    #   lanczos_num_iters=50)
    

    # initialize variables here
    k_curr = 4
    k_past = 0
    k = k_curr + k_past
    X = jnp.zeros((n, n))  # used to track primal solution
    y = jnp.zeros((n,))
    z = jnp.zeros((n,))

    # for interior point methods
    U = create_svec_matrix(k)

    # for quadratic subproblem solved by interior point method
    Q_base = create_Q_base(m, k, U)

    X, y, z, primal_obj, tr_X = specbm(
        X=X,
        y=y,
        z=z,
        primal_obj=0.0,
        tr_X=jnp.trace(X),
        n=n,
        m=m,
        trace_ub=trace_ub,
        C=C,
        C_innerprod=C_innerprod,
        C_add=C_add,
        C_matvec=C_matvec,
        A_operator=A_operator,
        A_operator_slim=A_operator_slim,
        A_adjoint=A_adjoint,
        A_adjoint_slim=A_adjoint_slim,
        Q_base=Q_base,
        U=U,
        b=b,
        rho=0.5,
        beta=0.25,
        k_curr=k_curr,
        k_past=k_past,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        eps=1e-4,
        max_iters=500,
        lanczos_num_iters=100)

    embed()
    exit()