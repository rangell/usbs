from functools import partial
import jax
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from jax._src.typing import Array
import numpy as np
from scipy.sparse import coo_matrix   # type: ignore
from typing import Callable, Tuple


@partial(jax.jit, static_argnames=["m"])
def apply_A_operator_slim(m: int, A_data: Array, A_indices: Array, u: Array) -> Array:
    outvec = jnp.zeros((m,))
    outvec = outvec.at[A_indices[:,0]].add(
        A_data * u.at[A_indices[:,1]].get() * u.at[A_indices[:,2]].get())
    return outvec


@partial(jax.jit, static_argnames=["n"])
def apply_A_adjoint_slim(n: int, A_data: Array, A_indices: Array, z: Array, u: Array) -> Array:
    outvec = jnp.zeros((n,))
    outvec = outvec.at[A_indices[:,1]].add(
        A_data * z.at[A_indices[:,0]].get() * u.at[A_indices[:,2]].get())
    return outvec


@partial(jax.jit, static_argnames=["m"])
def apply_A_operator_batched(m: int, A_data: Array, A_indices: Array, vecs: Array) -> Array:
    return jnp.sum(
        jax.vmap(apply_A_operator_slim, (None, None, None, 1), 1)(m, A_data, A_indices, vecs),
        axis=1)


@partial(jax.jit, static_argnames=["n"])
def apply_A_adjoint_batched(n: int, A_data: Array, A_indices: Array, z: Array, vecs: Array) -> Array:
    return jax.vmap(
        apply_A_adjoint_slim, (None, None, None, None, 1), 1)(n, A_data, A_indices, z, vecs)


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


@partial(jax.jit, static_argnames=["m", "k"])
def create_Q_base(m: int, k: int, U: BCOO, A_data: Array, A_indices: Array, V: Array) -> Array:
    base_tensor = jnp.zeros((m, k, k))
    base_tensor = base_tensor.at[A_indices[:, 0]].add(
        A_data.reshape(-1, 1, 1)
        * jax.lax.batch_matmul(V.at[A_indices[:, 1]].get().reshape(-1, k, 1),
                               V.at[A_indices[:, 2]].get().reshape(-1, 1, k)))
    flat_base_tensor = base_tensor.reshape(m, k**2).T
    svec_proj = U @ flat_base_tensor
    svec_dim_size = int(k*(k+1)/2)
    expanded_mx = svec_proj.reshape(svec_dim_size, 1, m) * svec_proj.reshape(1, svec_dim_size, m)
    final_mx = jnp.sum(expanded_mx, axis=-1)
    return final_mx


@jax.jit
def reconstruct_from_sketch(
    Omega: Array,
    P: Array,
    approx_eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    n = Omega.shape[0]
    rho = jnp.sqrt(n) * approx_eps * jnp.linalg.norm(P, ord=2)
    P_rho = P + rho * Omega
    B = Omega.T @ P_rho
    B = 0.5 * (B + B.T)
    L = jnp.linalg.cholesky(B)
    E, Rho, _ = jnp.linalg.svd(
        jnp.linalg.lstsq(L, P_rho.T, rcond=-1)[0].T,
        full_matrices=False,  # this compresses the output to be rank `R`
    )
    Lambda = jnp.clip(Rho ** 2 - rho, 0, np.inf)
    return E, Lambda