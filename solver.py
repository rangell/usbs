from collections import namedtuple
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
import numpy as np
import scipy  # type: ignore
from scipy.linalg import eigh_tridiagonal  # type: ignore 
from typing import Any, Callable, Tuple

from IPython import embed


# TODO: restructure to handle linear operator
# NOTE: The storage efficient version of this is NOT accurate
@partial(jax.jit,
         static_argnames=["C_matvec", "A_adjoint_slim", "proj_K", "n", "k", "num_iters", "eps"])
def approx_grad_k_min_eigen(
    C_matvec: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    proj_K: Callable[[Array], Array],
    y: Array,
    z: Array,
    beta: float,
    n: int,
    k: int,
    num_iters: int,
    eps: float,
    rng: Array
) -> Tuple[Array, Array]:

    # cache this since it will not change 
    adjoint_left_vec = y + beta*(z - proj_K(z + (y / beta)))

    TriDiagStateStruct = namedtuple("TriDiagStateStruct", ["t", "V", "diag", "off_diag"])

    def tri_diag_cond_func(state: TriDiagStateStruct) -> bool:
        #return jnp.logical_and(
        #    jnp.logical_or(state.off_diag[state.t] > eps, state.t == 1), state.t <= num_iters)
        return state.t <= num_iters  # Don't worry about eps for now, we will come back if we need it.

    def tri_diag_body_func(state: TriDiagStateStruct) -> TriDiagStateStruct:
        V = state.V
        diag = state.diag
        off_diag = state.off_diag
        transformed_v = C_matvec(V[state.t]) + A_adjoint_slim(adjoint_left_vec, V[state.t])
        transformed_v = transformed_v - (off_diag[state.t] * V[state.t-1]) # heed the off_diag index
        diag = diag.at[state.t].set(jnp.dot(V[state.t], transformed_v))
        v_next = transformed_v - (diag[state.t] * V[state.t])  

        # full reorthogonalization here
        v_next = lax.fori_loop(
            1, state.t+1, lambda i, vec: vec - (jnp.dot(vec, V[i]) * V[i]), v_next)

        off_diag = off_diag.at[state.t+1].set(jnp.linalg.norm(v_next))
        v_next /= off_diag[state.t+1]
        V = V.at[state.t+1].set(v_next)
        return TriDiagStateStruct(
            t=state.t+1, V=V, diag=diag, off_diag=off_diag
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

    final_state = lax.while_loop(tri_diag_cond_func, tri_diag_body_func, init_state)

    # TODO: implement Krylov subspace found prematurely:
    #   replace extra diags with `jnp.inf` and off_diags with `0`?
    # NOTE: this will not be needed until we change `tri_diag_cond_func` to reincorporate `eps`.

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
            max_it = 5  # Taken from LAPACK xSTEIN.
            min_norm_growth = 0.1
            norm_growth_factor = 1 + min_norm_growth
            # We stop the inverse iteration when we reach the maximum number of
            # iterations or the norm growths is less than 10%.
            return jnp.logical_and(
                jnp.less(i, max_it),
                jnp.any(
                    jnp.greater_equal(
                        jnp.real(nrm_v),
                        jnp.real(norm_growth_factor * nrm_v_old))))

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
        
        _, v, _, _ = lax.while_loop(
            continue_iteration, inverse_iteration_step, (0, v0, norm_v0, zero_norm))
        return v

    # Compute eigenvector for tridiagonal matrix
    tridiag_eigvecs = tridiag_eigvecs(diag, off_diag, min_k_eigvals)
    min_k_eigvecs = jnp.transpose(tridiag_eigvecs @ V)

    return min_k_eigvals, min_k_eigvecs


# don't have to jit this function? just jaxpr since it's only called once? YES
def cgal(
    n: int,
    m: int,
    trace_ub: float,
    C_innerprod: Callable[[Array], float],
    C_add: Callable[[Array], Array],
    C_matvec: Callable[[Array], Array],
    A_operator: Callable[[Array], Array],
    A_operator_slim: Callable[[Array], Array],
    A_adjoint: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    proj_K: Callable[[Array], Array],
    beta: float,
    SCALE_C: float,
    SCALE_X: float,
    eps: float,
    max_iters: int
) -> Tuple[Array, Array]:

    StateStruct = namedtuple("StateStruct", ["t", "X", "y", "obj_gap", "infeas_gap"])

    @jax.jit
    def cond_func(state: StateStruct) -> bool:
        return jnp.logical_and(
            jnp.logical_or(state.obj_gap > eps, state.infeas_gap > eps),
            state.t < max_iters)

    #@jax.jit
    def body_func(state: StateStruct) -> StateStruct:
        z = A_operator(state.X)
        b = proj_K(z + (state.y / beta))
        grad = C_add(A_adjoint(state.y + beta*(z - b)))
        eigvals0, eigvecs0 = jnp.linalg.eigh(grad)

        eigvals, eigvecs = approx_grad_k_min_eigen(
            C_matvec=C_matvec,
            A_adjoint_slim=A_adjoint_slim,
            proj_K=proj_K,
            y=state.y,
            z=z,
            beta=beta,
            n=n,
            k=2,
            num_iters=100,
            eps=eps,
            rng=jax.random.PRNGKey(0))

        embed()
        exit()

        # TODO: report eigval gap here!
        min_eigval = eigvals[0]
        min_eigvec = eigvecs[:, 0:1]  # gives the right shape
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

    state1 = body_func(init_state)

    embed()
    exit()

    final_state = lax.while_loop(cond_func, body_func, init_state)


    return final_state.X, final_state.y