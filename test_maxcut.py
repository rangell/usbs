import cvxpy as cp
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
import numpy as np
from pathlib import Path
import pickle
import scipy  # type: ignore
from scipy.io import loadmat  # type: ignore
from scipy.sparse import csc_matrix  # type: ignore
from typing import Any, Callable

import solver

from IPython import embed


def create_C_innerprod(C: csc_matrix) -> Callable[[Array], float]:
    indptr = jnp.array(C.indptr)
    indices = jnp.array(C.indices)
    data = jnp.array(C.data)
    @jax.jit
    def C_innerprod(X: Array) -> float:
        sum = 0.0
        inner_loop = lambda j, init_sum : lax.fori_loop(
            indptr[j],
            indptr[j+1],
            lambda offset, partial_sum: partial_sum + (data[offset] * X[indices[offset], j]),
            init_sum)
        sum = lax.fori_loop(0, indptr.size-1, inner_loop, sum)
        return sum
    return C_innerprod


def create_C_add(C: csc_matrix) -> Callable[[Array], Array]:
    indptr = jnp.array(C.indptr)
    indices = jnp.array(C.indices)
    data = jnp.array(C.data)
    @jax.jit
    def C_add(X: Array) -> Array:
        Y = jnp.copy(X)
        inner_loop = lambda j, init_Y: lax.fori_loop(
            indptr[j],
            indptr[j+1],
            lambda offset, partial_Y: partial_Y.at[indices[offset], j].add(data[offset]),
            init_Y)
        Y = lax.fori_loop(0, indptr.size-1, inner_loop, Y)
        return Y
    return C_add


def create_C_matvec(C: csc_matrix) -> Callable[[Array], Array]:
    indptr = jnp.array(C.indptr)
    indices = jnp.array(C.indices)
    data = jnp.array(C.data)
    n = C.shape[0]
    @jax.jit
    def C_matvec(u: Array) -> Array:
        v = jnp.zeros((n,))
        inner_loop = lambda j, init_v: lax.fori_loop(
            indptr[j],
            indptr[j+1],
            lambda offset, partial_v: partial_v.at[indices[offset]].add(u[j] * data[offset]),
            init_v)
        v = lax.fori_loop(0, indptr.size-1, inner_loop, v)
        return v
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
        return np.ones((n,)) * SCALE_X
    return proj_K


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
    np.random.seed(0)
    MAT_PATH = "./data/maxcut/Gset/G1.mat"
    problem = loadmat(MAT_PATH)
    C = problem["Problem"][0][0][1]
    n = C.shape[0]

    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = -0.25*C
    C = C.tocsc()

    SCALE_C = 1.0 / scipy.sparse.linalg.norm(C, ord="fro") 
    SCALE_X = 1.0 / n

    scs_soln_cache = str(Path(MAT_PATH).with_suffix("")) + "_scs_soln.pkl"
    if Path(scs_soln_cache).is_file():
        with open(scs_soln_cache, "rb") as f_in:
            X_scs = pickle.load(f_in)
    else:
        X_scs = solve_scs(C)
        with open(scs_soln_cache, "wb") as f_out:
            pickle.dump(X_scs, f_out)

    scaled_C = C * SCALE_C
    C_innerprod = create_C_innerprod(scaled_C)
    C_add = create_C_add(scaled_C)
    C_matvec = create_C_matvec(scaled_C)
    A_operator = create_A_operator()
    A_operator_slim = create_A_operator_slim()
    A_adjoint = create_A_adjoint(n)
    A_adjoint_slim = create_A_adjoint_slim()
    proj_K = create_proj_K(n, SCALE_X)

    #rng = jax.random.PRNGKey(0)
    #M = jax.random.normal(rng, shape=(100, 100))
    #Q, _ = jnp.linalg.qr(M) # columns of Q are orthonormal
    #eigvals = jnp.ceil(jnp.arange(1, 32, 0.3))[:100].reshape(1, -1)
    ##eigvals = jnp.ceil(jnp.arange(1, 101, 1))[:100].reshape(1, -1)
    #M = (Q * eigvals) @ Q.T

    #eigvals, eigvecs = solver.approx_k_min_eigen(
    #    M = lambda v: M @ v, n=100, k=6, num_iters=50, eps=1e-6, rng=rng)

    X, y = solver.cgal(
       n=n,
       m=n,
       trace_ub=1.0,
       C_innerprod=C_innerprod,
       C_add=C_add,
       C_matvec=C_matvec,
       A_operator=A_operator,
       A_operator_slim=A_operator_slim,
       A_adjoint=A_adjoint,
       A_adjoint_slim=A_adjoint_slim,
       proj_K=proj_K,
       beta=1.0,
       SCALE_C=SCALE_C,
       SCALE_X=SCALE_X,
       eps=1.0,
       max_iters=1e6)
