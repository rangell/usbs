from collections import namedtuple
import jax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
from jax.experimental import sparse
import jax.numpy as jnp
from typing import Tuple


SDPState = namedtuple("SDPState",
                      ["C",
                       "A_indices",
                       "A_data",
                       "b",
                       "b_ineq_mask",
                       "X",
                       "P",
                       "Omega",
                       "y",
                       "z",
                       "tr_X",
                       "primal_obj",
                       "SCALE_C",
                       "SCALE_X",
                       "SCALE_A"])


def scale_sdp_state(sdp_state: SDPState) -> SDPState:
    scaled_C = BCOO((sdp_state.C.data * sdp_state.SCALE_C, sdp_state.C.indices),
                    shape=sdp_state.C.shape)
    scaled_A_data = sdp_state.A_data * sdp_state.SCALE_A.at[sdp_state.A_indices[:,0]].get()
    scaled_b = sdp_state.b * sdp_state.SCALE_X * sdp_state.SCALE_A

    scaled_X = sdp_state.X
    scaled_P = sdp_state.P
    if sdp_state.X is not None:
        scaled_X = sdp_state.X * sdp_state.SCALE_X
    if sdp_state.P is not None:
        scaled_P = sdp_state.P * sdp_state.SCALE_X

    scaled_z = sdp_state.z * sdp_state.SCALE_A * sdp_state.SCALE_X
    scaled_tr_X = sdp_state.tr_X * sdp_state.SCALE_X
    scaled_primal_obj = sdp_state.primal_obj * sdp_state.SCALE_X * sdp_state.SCALE_C

    return SDPState(
        C=scaled_C,
        A_indices=sdp_state.A_indices,
        A_data=scaled_A_data,
        b=scaled_b,
        b_ineq_mask=sdp_state.b_ineq_mask,
        X=scaled_X,
        P=scaled_P,
        Omega=sdp_state.Omega,
        y=sdp_state.y,
        z=scaled_z,
        tr_X=scaled_tr_X,
        primal_obj=scaled_primal_obj,
        SCALE_C=sdp_state.SCALE_C,
        SCALE_X=sdp_state.SCALE_X,
        SCALE_A=sdp_state.SCALE_A)


def unscale_sdp_state(sdp_state: SDPState) -> SDPState:
    unscaled_C = BCOO((sdp_state.C.data / sdp_state.SCALE_C, sdp_state.C.indices),
                    shape=sdp_state.C.shape)
    unscaled_A_data = sdp_state.A_data / sdp_state.SCALE_A.at[sdp_state.A_indices[:,0]].get()
    unscaled_b = (sdp_state.b / sdp_state.SCALE_A) / sdp_state.SCALE_X

    unscaled_X = sdp_state.X
    unscaled_P = sdp_state.P
    if sdp_state.X is not None:
        unscaled_X = sdp_state.X / sdp_state.SCALE_X
    if sdp_state.P is not None:
        unscaled_P = sdp_state.P / sdp_state.SCALE_X

    unscaled_z = (sdp_state.z / sdp_state.SCALE_A ) / sdp_state.SCALE_X
    unscaled_tr_X = sdp_state.tr_X / sdp_state.SCALE_X
    unscaled_primal_obj = sdp_state.primal_obj / (sdp_state.SCALE_X * sdp_state.SCALE_C)

    return SDPState(
        C=unscaled_C,
        A_indices=sdp_state.A_indices,
        A_data=unscaled_A_data,
        b=unscaled_b,
        b_ineq_mask=sdp_state.b_ineq_mask,
        X=unscaled_X,
        P=unscaled_P,
        Omega=sdp_state.Omega,
        y=sdp_state.y,
        z=unscaled_z,
        tr_X=unscaled_tr_X,
        primal_obj=unscaled_primal_obj,
        SCALE_C=sdp_state.SCALE_C,
        SCALE_X=sdp_state.SCALE_X,
        SCALE_A=sdp_state.SCALE_A)


@jax.jit
def reconstruct_from_sketch(
    Omega: Array,
    P: Array,
    approx_eps: float = 1e-6
) -> Tuple[Array, Array]:
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
    Lambda = jnp.clip(Rho ** 2 - rho, 0, jnp.inf)
    return E, Lambda


def apply_A_operator_mx(n: int, m: int, A_data: Array, A_indices: Array, X: Array) -> Array:
    A = BCOO((A_data, A_indices), shape=(m, n, n))
    return sparse.bcoo_reduce_sum(A * X[None, :, :], axes=[1,2]).todense()