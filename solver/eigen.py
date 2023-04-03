from collections import namedtuple
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
from typing import Any, Callable, Tuple

from IPython import embed
@partial(jax.jit, static_argnames=["C_matvec", "A_adjoint_slim", "n", "k", "num_iters"])
def approx_grad_k_min_eigen(
    C_matvec: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    adjoint_left_vec: Array,
    n: int,
    k: int,
    num_iters: Array,
    rng: Array
) -> Tuple[Array, Array]:

    TriDiagStateStruct = namedtuple("TriDiagStateStruct", ["t", "V", "diag", "off_diag"])

    def tri_diag_body_func(t: int, state: TriDiagStateStruct) -> TriDiagStateStruct:
        V = state.V
        diag = state.diag
        off_diag = state.off_diag
        transformed_v = C_matvec(V[t]) + A_adjoint_slim(adjoint_left_vec, V[t])
        transformed_v = transformed_v - (off_diag[t] * V[t-1]) # heed the off_diag index
        diag = diag.at[t].set(jnp.dot(V[t], transformed_v))
        v_next = transformed_v - (diag[t] * V[t])  

        # full reorthogonalization here
        v_next -= jnp.sum((V @ v_next).reshape(-1, 1) * V, axis=0)

        off_diag = off_diag.at[t+1].set(jnp.linalg.norm(v_next))
        v_next /= off_diag[t+1]
        V = V.at[t+1].set(v_next)
        return TriDiagStateStruct(
            t=t+1, V=V, diag=diag, off_diag=off_diag
        )

    v_1 = jax.random.normal(rng, shape=(n,))
    v_1 = v_1 / jnp.linalg.norm(v_1)

    # (*) dimension hacking to make it easier for jax
    V = jnp.zeros((num_iters+2, n)) 
    V = V.at[1].set(v_1)
    init_state = TriDiagStateStruct(
        t=1,
        V=V,
        diag=jnp.zeros((num_iters+1,)),
        off_diag=jnp.zeros((num_iters+2,)))

    final_state = lax.fori_loop(1, num_iters+1, tri_diag_body_func, init_state)

    # remove dimension hacking initiated at (*)
    V = final_state.V[1:-1,:]
    diag = final_state.diag[1:]
    off_diag = final_state.off_diag[2:-1]

    # Compute eigenvectors from tridiagonal matrix
    min_k_eigvals = jax.scipy.linalg.eigh_tridiagonal(
        diag,
        off_diag,
        select="i",
        select_range=(0, k-1),
        eigvals_only=True)

    # Since jax only implements `eigh_tridiagonal` for eigenvalues, we need to compute
    # eigenvectors for ourselves. Below is adapted from tensorflow implementation:
    # https://github.com/tensorflow/tensorflow/blob/c1a369e066d94418ee4f6d8aeaf7fbe086441fc0/tensorflow/python/ops/linalg/linalg_impl.py#L1460-L1585
    @jax.named_scope("tridiag_eigvecs")
    def tridiag_eigvecs(diag, off_diag, eigvals):
        k = eigvals.size
        n = diag.size

        # We perform inverse iteration for all eigenvectors in parallel,
        # starting from a random set of vectors, until all have converged.
        v0 = jax.random.normal(rng, shape=(k, n), dtype=off_diag.dtype)
        norm_v0 = jnp.linalg.norm(v0, axis=1)
        v0 = v0 / norm_v0.reshape(-1, 1)
        zero_norm = jnp.zeros(norm_v0.shape, dtype=norm_v0.dtype)

        # Replicate alpha-eigvals(ik) and beta across the k eigenvectors so we
        # can solve the k systems
        #    [T - eigvals(i)*eye(n)] x_i = r_i
        # simultaneously using the batching mechanism.
        eigvals_cast = eigvals.astype(dtype=off_diag.dtype)
        off_diag = jnp.tile(off_diag.reshape(1, -1), [k, 1])
        d = (diag.reshape(1, -1) - eigvals_cast.reshape(-1, 1))
        dl = jnp.concatenate([jnp.zeros((k, 1)), jnp.conj(off_diag)], axis=1)
        du = jnp.concatenate([off_diag, jnp.zeros((k, 1))], axis=1)

        def continue_iteration(state: Tuple[int, Array, Array, Array]):
            i, _, nrm_v, nrm_v_old = state
            min_norm_growth = 0.1
            norm_growth_factor = 1 + min_norm_growth
            # We stop the inverse iteration when we reach the maximum number of
            # iterations or the norm growths is less than 10%.
            return jnp.any(jnp.greater_equal(jnp.real(nrm_v), jnp.real(norm_growth_factor * nrm_v_old)))

        def inverse_iteration_step(state: Tuple[int, Array, Array, Array]):
            i, v, nrm_v, nrm_v_old = state
            v = lax.fori_loop(
                lower=0, 
                upper=k,
                body_fun=lambda i, v: v.at[i].set(
                    lax.linalg.tridiagonal_solve(
                        dl=dl[i], d=d[i], du=du[i], b=v[i].reshape(-1, 1)).reshape(-1,)),
                init_val=v)
            nrm_v_old = nrm_v
            nrm_v = jnp.linalg.norm(v, axis=1)
            v = v / nrm_v.reshape(-1, 1)
            # orthogonalize for numerical stability
            q, _ = jnp.linalg.qr(jnp.transpose(v))
            v = jnp.transpose(q)
            return i+1, v, nrm_v, nrm_v_old

        # `max_steps` taken from LAPACK xSTEIN.
        _, v, _, _ = bounded_while_loop(
            continue_iteration, inverse_iteration_step, (0, v0, norm_v0, zero_norm), max_steps=5)
        return v

    # Compute eigenvector for tridiagonal matrix
    tridiag_eigvecs = tridiag_eigvecs(diag, off_diag, min_k_eigvals)
    min_k_eigvecs = jnp.transpose(tridiag_eigvecs @ V)

    return min_k_eigvals, min_k_eigvecs