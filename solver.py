from collections import namedtuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import scipy  # type: ignore
from scipy.linalg import eigh_tridiagonal  # type: ignore 
from typing import Any, Callable, Tuple

from IPython import embed


# TODO: implement storage efficient version?
def approx_k_min_eigen(
    M: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    k: int,
    num_iters: int,
    eps: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    V = np.empty((num_iters, n))
    omegas = np.empty((num_iters,))
    rhos = np.empty((num_iters - 1,))

    v_0 = np.random.normal(size=(n,))
    V[0] = v_0 / np.linalg.norm(v_0)

    for i in range(num_iters):
        transformed_v = M(V[i])
        omegas[i] = np.dot(V[i], transformed_v)
        if i == num_iters - 1:
            break  # we have all we need
        V[i + 1] = transformed_v - (omegas[i] * V[i])
        if i > 0:
            V[i + 1] -= rhos[i - 1] * V[i - 1]
        rhos[i] = np.linalg.norm(V[i + 1])
        if rhos[i] < eps:
            break
        V[i + 1] = V[i + 1] / rhos[i]

        # TODO: perform full re-orthogonalization here?

    min_eigen_val, u = eigh_tridiagonal(
        omegas[: i + 1], rhos[:i], select="i", select_range=(0, 0)
    )
    min_eigen_vec = (u.T @ V[: i + 1])
    # renormalize for stability
    min_eigen_vec = min_eigen_vec / np.linalg.norm(min_eigen_vec)

    #max_eigen_val, u = eigh_tridiagonal(
    #    omegas[: i + 1], rhos[:i], select="i", select_range=(i, i)
    #)
    #max_eigen_vec = (u.T @ V[: i + 1]).squeeze()
    ## renormalize for stability
    #max_eigen_vec = max_eigen_vec / np.linalg.norm(max_eigen_vec)

    return min_eigen_val.squeeze(), min_eigen_vec.T

# don't have to jit this function? just jaxpr since it's only called once? YES
def cgal(
    n: int,
    m: int,
    trace_ub: float,
    C_innerprod: Callable[[jnp.ndarray], float],
    C_add: Callable[[jnp.ndarray], jnp.ndarray],
    C_matvec: Callable[[jnp.ndarray], jnp.ndarray],
    A_operator: Callable[[jnp.ndarray], jnp.ndarray],
    A_operator_slim: Callable[[jnp.ndarray], jnp.ndarray],
    A_adjoint: Callable[[jnp.ndarray], jnp.ndarray],
    A_adjoint_slim: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    proj_K: Callable[[jnp.ndarray], jnp.ndarray],
    beta: float,
    SCALE_C: float,
    SCALE_X: float,
    eps: float,
    max_iters: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    StateStruct = namedtuple(
        "StateStruct", 
        ["t", "X", "y", "obj_gap", "infeas_gap"])

    @jax.jit
    def stop_func(state: StateStruct) -> bool:
        # hacky jax-compatible implementation of the following predicate (to continue optimizing):
        #   (obj_gap > eps or infeas_gap > eps) and state.t < max_iters
        predicates = jnp.array([state.obj_gap > eps, state.infeas_gap > eps], dtype=jnp.uint8)
        jax.debug.print("t: {t} - predicates 1: {predicates}", t=state.t, predicates=predicates)
        predicates = jnp.array([jnp.sum(predicates) > 0, state.t < max_iters], dtype=jnp.uint8)
        jax.debug.print("t: {t} - predicates 2: {predicates}", t=state.t, predicates=predicates)
        return jnp.sum(predicates) == 2

    @jax.jit
    def body_func(state: StateStruct) -> StateStruct:
        z = A_operator(state.X)
        b = proj_K(z + (state.y / beta))
        grad = C_add(A_adjoint(state.y + beta*(z - b)))
        eigvals, eigvecs = jnp.linalg.eigh(grad)
        # TODO: report eigval gap here!
        min_eigval = eigvals[0]
        min_eigvec = eigvecs.at[:, 0:1].get()  # gives the right shape
        X_update_dir = min_eigvec @ min_eigvec.T
        eta = 2.0 / (state.t + 2.0)   # just use the standard CGAL step-size for now
        surrogate_dual_gap = jnp.trace(grad @ (state.X - X_update_dir))
        obj_gap = surrogate_dual_gap - jnp.dot(state.y, z - b) - 0.5*beta*jnp.linalg.norm(z - b)**2
        obj_gap = obj_gap / (SCALE_C * SCALE_X)
        infeas_gap = jnp.max(jnp.abs(z - proj_K(z))) / SCALE_X
        jax.debug.print("t: {t} - obj_val: {obj_val} - obj_gap: {obj_gap} - infeas_gap: {infeas_gap}",
                        t=state.t,
                        obj_val=C_innerprod(state.X) / (SCALE_C * SCALE_X),
                        obj_gap=obj_gap,
                        infeas_gap=infeas_gap)
        X_next = (1-eta)*state.X + eta*X_update_dir
        z_next = A_operator(X_next)
        y_next = state.y + (z_next - proj_K(z_next + (state.y / beta)))
        return StateStruct(
            t=state.t+1,
            X=X_next,
            y=y_next,
            obj_gap=obj_gap,
            infeas_gap=infeas_gap)



    init_state = StateStruct(
        t=0,
        X=jnp.zeros((n, n)) * SCALE_X,
        y=jnp.zeros((m,)),
        obj_gap=1.1*eps,
        infeas_gap=1.1*eps)

    final_state = lax.while_loop(stop_func, body_func, init_state)

    embed()
    exit()

    return final_state.X, final_state.y