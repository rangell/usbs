from collections import namedtuple
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import scipy  # type: ignore
from scipy.linalg import eigh_tridiagonal  # type: ignore 
from typing import Any, Callable, Tuple

from IPython import embed


def approx_min_eigen(
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
    SCALE_C: float,
    SCALE_X: float,
    eps: float,
    max_iters: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:

    StateStruct = namedtuple(
        "StateStruct", 
        ["t", "X", "X_next", "y_next", "grad", "X_update_dir"])

    @jax.jit
    def stop_func(state: StateStruct) -> bool:
        obj_gap = jnp.trace(state.grad @ (state.X - state.X_update_dir)) / (SCALE_X * SCALE_C)
        z_next = A_operator(state.X_next)
        infeas_gap = jnp.max(jnp.abs(z_next - proj_K(z_next)) / SCALE_X)
        return (obj_gap > eps and infeas_gap > eps) or state.t == 0 or state.t < max_iters

    @jax.jit
    def body_func(state: StateStruct) -> StateStruct:
        X = state.X_next
        y = state.y_next
        z = A_operator(X)
        grad = C_add(A_adjoint(y + z - proj_K(z + y)))
        eigvals, eigvecs = jnp.linalg.eigh(grad)
        # TODO: report eigval gap here!
        min_eigvec = eigvecs.at[:, 0:1].get()  # gives the right shape
        X_update_dir = min_eigvec @ min_eigvec.T
        eta = 2.0 / (state.t + 2.0)   # just use the standard CGAL step-size for now
        X_next = (1-eta)*X + eta*X_update_dir
        z_next = A_operator(X_next)
        y_next = y + (z_next - proj_K(z_next + y))
        return StateStruct(
            t=state.t+1,
            X=X,
            X_next=X_next,
            y_next=y_next,
            grad=grad,
            X_update_dir=X_update_dir)

    init_state = StateStruct(
        t=0,
        X=jnp.empty((n, n)),
        X_next=jnp.ones((n, n)) * trace_ub / n,
        y_next=jnp.zeros((m,)),
        grad=jnp.empty((n, n)),
        X_update_dir=jnp.empty((n, n)))

    state1 = body_func(init_state)
    state2 = body_func(state1)

    embed()
    exit()

    z = A_operator(X)
    for t in range(max_iters):
        eta = 2.0 / (t + 2.0)
        z = A_operator(X)
        grad = C_add(A_adjoint(y) + A_adjoint(z - proj_K(z + y)))
        min_eigval, min_eigvec = scipy.sparse.linalg.eigsh(grad, k=1, which="SA")

        num_iters = int((np.ceil((t + 1) ** (1 / 4)) * np.log(n)))
        num_iters = 200
        min_eigval2, min_eigvec2 = approx_min_eigen(lambda v: grad @ v, n, num_iters, eps)

        embed()
        exit()

        X_update_dir = min_eigvec @ min_eigvec.T
        X_next = (1-eta)*X + eta*X_update_dir
        z = A_operator(X)
        y_next = y + (z - proj_K(z + y))
        obj_gap = np.trace(grad @ (X - X_update_dir)) / (SCALE_X * SCALE_C)
        infeas_gap = np.max(np.abs(y - y_next) / SCALE_X)
        print("CGAL t: ", t,
              " obj_gap: ", obj_gap,
              " infeas: ", infeas_gap,
              " obj_val: ", C_innerprod(X_next))
        if obj_gap < eps and infeas_gap < eps:
            break
        X = X_next
        y = y_next
    return X, y