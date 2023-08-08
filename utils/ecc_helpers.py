from functools import partial
import jax
from jax import lax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import numba as nb
import numpy as np
import pickle
from scipy.spatial.distance import pdist, squareform  # type: ignore
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
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

    # constraint: diagonal of X is all 1's
    range_n = jnp.arange(n)[:, None]
    A_indices = jnp.hstack(3*[range_n])
    A_data = jnp.ones((n,))
    b = jnp.ones((n,))
    b_ineq_mask = jnp.zeros((n,))

    # constraint: objective-relevant entries of X >= 0, written as -X <= 0
    triu_indices_mask = (C.indices[:, 0] <= C.indices[:, 1])
    constraint_indices = b.shape[0] + jnp.arange(jnp.sum(triu_indices_mask))
    constraint_triples = jnp.concatenate(
        [constraint_indices[:, None], C.indices[triu_indices_mask]], axis=1)
    constraint_triples = jnp.concatenate(
        [constraint_triples, constraint_triples[:, [0, 2, 1]]], axis=0)
    A_indices = jnp.concatenate([A_indices, constraint_triples], axis=0)
    A_data = jnp.concatenate([A_data, jnp.full((constraint_triples.shape[0],), -1.0)], axis=0)
    b = jnp.concatenate([b, jnp.full((constraint_indices.shape[0],), 0.0)], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.full((constraint_indices.shape[0],), 1.0)], axis=0)

    return A_data, A_indices, b, b_ineq_mask


def initialize_state(C: BCOO, sketch_dim: int) -> SDPState:
    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    SCALE_X = 1.0 / float(n)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False)[0])
    SCALE_A /= norm_A

    #SCALE_X = 1.0
    #SCALE_C = 1.0
    #SCALE_A = jnp.ones_like(b)

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

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state