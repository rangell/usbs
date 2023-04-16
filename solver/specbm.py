from collections import namedtuple
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
from typing import Any, Callable, Tuple

from solver.eigen import approx_grad_k_min_eigen

from IPython import embed


def solve_subproblem(
    C_matvec: Callable[[Array], Array],
    A_operator_slim: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    b: Array,
    trace_ub: float,
    rho: float,
    obj_val: Array,
    z: Array,
    y: Array,
    V: Array,
    k: int,
    apgd_step_size: float,
    apgd_max_iters: int,
    apgd_eps: float
) -> Tuple[Array, Array, Array]:
    C_matmat = jax.vmap(C_matvec, 1, 1)
    A_adjoint_batched = jax.vmap(A_adjoint_slim, (None, 1), 1)
    A_operator_batched = jax.vmap(A_operator_slim, 1, 1)

    # spectral line search
    APGDState = namedtuple(
        "APGDState",
        ["i", "eta_curr", "eta_past", "S_curr", "S_past", "max_value_change"])

    @jax.jit
    def apgd(apgd_state: APGDState) -> APGDState:
        momentum = apgd_state.i / (apgd_state.i + 3)  # set this to 0.0 for standard PGD
        S = apgd_state.S_curr +  momentum * (apgd_state.S_curr - apgd_state.S_past)
        eta = apgd_state.eta_curr + momentum * (apgd_state.eta_curr - apgd_state.eta_past)
        S_eigvals, S_eigvecs = jnp.linalg.eigh(S)

        # for numerical stability, make sure all eigvals are >= 0
        S_eigvals = jnp.where(S_eigvals < 0, 0, S_eigvals)

        # compute gradients
        VSV_T_factor = (V @ S_eigvecs) * jnp.sqrt(S_eigvals).reshape(1, -1)
        A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)
        grad_S = (V.T @ C_matmat(V)
                    + V.T @ A_adjoint_batched(y, V)
                    + (1.0/rho) * V.T @ A_adjoint_batched((eta*z) + A_operator_VSV_T - b, V))
        grad_eta = (obj_val
                    + jnp.dot(y, z)
                    + (1.0/rho) * eta * jnp.linalg.norm(z)**2
                    + (1.0/rho) * jnp.dot(z, A_operator_VSV_T - b))

        # compute unprojected steps
        S_unproj = S - (apgd_step_size * grad_S)
        eta_unproj = eta - (apgd_step_size * grad_eta)

        S_unproj_eigvals, S_eigvecs = jnp.linalg.eigh(S_unproj)
        trace_vals = jnp.append(S_unproj_eigvals / trace_ub, eta_unproj)

        def proj_simplex(unsorted_vals: Array) -> Array:
            inv_sort_indices = jnp.argsort(jnp.argsort(unsorted_vals))
            sorted_vals = jnp.sort(unsorted_vals)
            descend_vals = jnp.flip(sorted_vals)
            weighted_vals = (descend_vals
                            + (1.0 / jnp.arange(1, len(descend_vals)+1))
                                * (1 - jnp.cumsum(descend_vals)))
            idx = jnp.sum(weighted_vals > 0) - 1
            offset = weighted_vals[idx] - descend_vals[idx]
            proj_descend_vals = descend_vals + offset
            proj_descend_vals = proj_descend_vals * (proj_descend_vals > 0)
            proj_unsorted_vals = jnp.flip(proj_descend_vals)[inv_sort_indices]
            return proj_unsorted_vals

        # check to see if we do not need to project onto the simplex
        no_projection_needed = jnp.logical_and(jnp.sum(trace_vals) < 1,
                                               jnp.all(trace_vals > -1e-6))
        proj_trace_vals = lax.cond(
            no_projection_needed,
            lambda arr: arr,
            lambda arr: proj_simplex(arr),
            trace_vals)

        # get projected next step values
        eta_next = proj_trace_vals[-1]
        proj_S_eigvals = trace_ub * proj_trace_vals[:-1]
        S_next = (S_eigvecs * proj_S_eigvals.reshape(1, -1)) @ S_eigvecs.T
        max_value_change = jnp.max(
            jnp.append(jnp.abs(apgd_state.S_curr - S_next).reshape(-1,),
                        jnp.abs(apgd_state.eta_curr - eta_next)))
        return APGDState(
            i=apgd_state.i+1,
            eta_curr=eta_next,
            eta_past=apgd_state.eta_curr,
            S_curr=S_next,
            S_past=apgd_state.S_curr,
            max_value_change=max_value_change)

    #init_apgd_state = APGDState(
    #    i=0.0,
    #    eta_curr=jnp.array(0.0),
    #    eta_past=jnp.array(0.0),
    #    S_curr=jnp.eye(k)/k*trace_ub,
    #    S_past=jnp.zeros((k,k)),
    #    max_value_change=jnp.array(1.1*eps))
    init_apgd_state = APGDState(
        i=0.0,
        eta_curr=jnp.array(1.0),
        eta_past=jnp.array(0.0),
        S_curr=jnp.zeros((k,k)),
        S_past=jnp.zeros((k,k)),
        max_value_change=jnp.array(1.1*apgd_eps))

    final_apgd_state = bounded_while_loop(
        lambda apgd_state: apgd_state.max_value_change > apgd_eps,
        apgd, 
        init_apgd_state,
        max_steps=apgd_max_iters)


def specbm(
    X: Array,
    y: Array,
    z: Array,
    obj_val: float,
    V: Array,
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
    b: Array,
    rho: float,
    beta: float,
    k_curr: int,
    k_past: int,
    SCALE_C: float,
    SCALE_X: float,
    eps: float,
    max_iters: int,
    lanczos_num_iters: int,
    apgd_step_size: float,
    apgd_max_iters: int,
    apgd_eps: float
) -> Tuple[Array, Array]:

    # TODO: fix me
    return jnp.zeros(n, n), jnp.zeros((m,))
