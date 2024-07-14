from functools import partial

import jax
from jax import lax
from jax._src.typing import Array
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import numba as nb
import numpy as np
import scipy  # type: ignore
from scipy.spatial.distance import pdist, squareform  # type: ignore
from scipy.sparse import csc_matrix  # type:ignore
from typing import Any, Tuple

from solver.utils import apply_A_operator_batched
from utils.common import (SDPState,
                          scale_sdp_state,
                          unscale_sdp_state,
                          reconstruct_from_sketch,
                          apply_A_operator_mx)

from IPython import embed


def get_all_problem_data(C: BCOO) -> Tuple[BCOO, Array, Array, Array]:
    n = C.shape[0]
    range_n = jnp.arange(n)[:, None]
    A_indices = jnp.hstack(3*[range_n])
    A_data = jnp.ones((n,))
    b = jnp.ones((n,))
    b_ineq_mask = jnp.zeros((n,))
    return A_data, A_indices, b, b_ineq_mask


def initialize_state(C: csc_matrix, sketch_dim: int) -> SDPState:
    n = C.shape[0]
    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = 0.25*C
    C = C.tocsc()
    C = BCOO.from_scipy_sparse(C)

    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    m = b.shape[0]

    #SCALE_X = 1.0 / float(n)
    #SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to Frobenius norm
    SCALE_X = 1.0
    SCALE_C = 1.0
    SCALE_A = jnp.ones_like(b)

    if sketch_dim == -1:
        X = jnp.zeros((n, n))
        Omega = None
        P = None
    elif sketch_dim > 0:
        X = None
        Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, sketch_dim))
        P = jnp.zeros_like(Omega)
    else:
        raise ValueError("Invalid value for sketch_dim")

    y = jnp.zeros((m,))
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    print("SCALE_C: ", SCALE_C)
    print("SCALE_X: ", SCALE_X)
    print("min(SCALE_A): ", jnp.min(SCALE_A))
    print("max(SCALE_A): ", jnp.max(SCALE_A))

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


def get_implicit_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    assert sketch_dim == -1 or sketch_dim == old_sdp_state.Omega.shape[1]
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    n = C.shape[0]
    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = 0.25*C
    C = C.tocsc()
    C = BCOO.from_scipy_sparse(C)

    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    m = b.shape[0]

    X = old_sdp_state.X
    Omega = old_sdp_state.Omega
    P = old_sdp_state.P
    if old_sdp_state.X is not None:
        X = BCOO.fromdense(old_sdp_state.X)
        X = BCOO((X.data, X.indices), shape=(n, n)).todense()
    if old_sdp_state.P is not None:
        P = BCOO.fromdense(old_sdp_state.P)
        P = BCOO((P.data, P.indices), shape=(n, sketch_dim)).todense()
        Omega_pad = jax.random.normal(
            jax.random.PRNGKey(n), shape=(n - old_sdp_state.C.shape[0], sketch_dim))
        Omega = jnp.concatenate([old_sdp_state.Omega, Omega_pad], axis=0)
        
    y = jnp.zeros((m,)).at[jnp.arange(old_sdp_state.b.shape[0])].set(old_sdp_state.y)
    z = jnp.zeros((m,)).at[jnp.arange(old_sdp_state.b.shape[0])].set(old_sdp_state.z)

    old_diag_mask = (old_sdp_state.C.indices[:, 0] == old_sdp_state.C.indices[:, 1])
    diag_mask = (C.indices[:, 0] == C.indices[:, 1])
    old_C_diag = jnp.zeros((old_sdp_state.C.shape[0],)).at[
        old_sdp_state.C.indices[old_diag_mask][:, 0]].set(old_sdp_state.C.data[old_diag_mask])
    C_diag = jnp.zeros((C.shape[0],)).at[C.indices[diag_mask][:, 0]].set(C.data[diag_mask])
    primal_obj = old_sdp_state.primal_obj
    primal_obj -= jnp.dot(old_C_diag, old_sdp_state.z)
    primal_obj += jnp.dot(C_diag, z)

    SCALE_X = 1.0 / float(n)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to Frobenius norm
    SCALE_A = jnp.ones_like(b)

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=old_sdp_state.tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    print("SCALE_C: ", SCALE_C)
    print("SCALE_X: ", SCALE_X)
    print("min(SCALE_A): ", jnp.min(SCALE_A))
    print("max(SCALE_A): ", jnp.max(SCALE_A))

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


def get_explicit_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    assert sketch_dim == -1 or sketch_dim == old_sdp_state.Omega.shape[1]
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    n = C.shape[0]
    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = 0.25*C
    C = C.tocsc()
    C = BCOO.from_scipy_sparse(C)

    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    m = b.shape[0]

    X = old_sdp_state.X
    Omega = old_sdp_state.Omega
    P = old_sdp_state.P
    if old_sdp_state.X is not None:
        X = BCOO.fromdense(old_sdp_state.X)
        X = BCOO((X.data, X.indices), shape=(n, n)).todense()
        z = apply_A_operator_mx(n, m, A_data, A_indices, X) 
        tr_X = jnp.trace(X)
        primal_obj = jnp.trace(C @ X)
    if old_sdp_state.P is not None:
        Omega = jax.random.normal(jax.random.PRNGKey(n), shape=(n, sketch_dim))
        E, Lambda = reconstruct_from_sketch(old_sdp_state.Omega, old_sdp_state.P)
        tr_offset = (old_sdp_state.tr_X - jnp.sum(Lambda)) / Lambda.shape[0]
        Lambda_tr_correct = Lambda + tr_offset
        E = BCOO.fromdense(E)
        E = BCOO((E.data, E.indices), shape=Omega.shape).todense()
        sqrt_X_hat = E * jnp.sqrt(Lambda_tr_correct)[None, :]
        P = sqrt_X_hat @ (sqrt_X_hat.T @ Omega)
        z = apply_A_operator_batched(m, A_data, A_indices, sqrt_X_hat)
        tr_X = jnp.sum(Lambda_tr_correct)
        primal_obj = jnp.trace(sqrt_X_hat.T @ (C @ sqrt_X_hat))

    y = jnp.zeros((m,)).at[jnp.arange(old_sdp_state.b.shape[0])].set(old_sdp_state.y)

    SCALE_X = 1.0 / float(n)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to Frobenius norm
    SCALE_A = jnp.ones_like(b)

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    print("SCALE_C: ", SCALE_C)
    print("SCALE_X: ", SCALE_X)
    print("min(SCALE_A): ", jnp.min(SCALE_A))
    print("max(SCALE_A): ", jnp.max(SCALE_A))

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


def get_dual_only_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    assert sketch_dim == -1 or sketch_dim == old_sdp_state.Omega.shape[1]
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    n = C.shape[0]
    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = 0.25*C
    C = C.tocsc()
    C = BCOO.from_scipy_sparse(C)

    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    m = b.shape[0]

    SCALE_X = 1.0 / float(n)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to Frobenius norm
    SCALE_A = jnp.ones_like(b)

    if sketch_dim == -1:
        X = jnp.zeros((n, n))
        Omega = None
        P = None
    elif sketch_dim > 0:
        X = None
        Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, sketch_dim))
        P = jnp.zeros_like(Omega)
    else:
        raise ValueError("Invalid value for sketch_dim")

    y = jnp.zeros((m,)).at[jnp.arange(old_sdp_state.b.shape[0])].set(old_sdp_state.y)
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    print("SCALE_C: ", SCALE_C)
    print("SCALE_X: ", SCALE_X)
    print("min(SCALE_A): ", jnp.min(SCALE_A))
    print("max(SCALE_A): ", jnp.max(SCALE_A))

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


@partial(jax.jit, static_argnames=["callback_static_args"])
def compute_max_cut(
    P: Array,
    Omega: Array,
    callback_static_args: bytes,
    callback_nonstatic_args: Any
) -> float:
    C = callback_nonstatic_args 
    W, _ = reconstruct_from_sketch(Omega, P)
    W_bin = 2 * (W > 0).astype(float) - 1
    return jnp.max(jnp.diag(W_bin.T @ C @ W_bin))
