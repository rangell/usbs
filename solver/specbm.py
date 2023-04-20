from collections import namedtuple
import cvxpy as cp
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
from typing import Any, Callable, Tuple

from scipy.sparse import csc_matrix  # type: ignore

from solver.eigen import approx_grad_k_min_eigen

from IPython import embed


def solve_subproblem(
    C_matmat: Callable[[Array], Array],
    A_operator_batched: Callable[[Array], Array],
    A_adjoint_batched: Callable[[Array, Array], Array],
    b: Array,
    trace_ub: float,
    rho: float,
    primal_obj_val: Array,
    z: Array,
    y: Array,
    V: Array,
    k: int,
    apgd_step_size: float,
    apgd_max_iters: int,
    apgd_eps: float
) -> Tuple[Array, Array]:

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
        grad_eta = (primal_obj_val
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

    init_apgd_state = APGDState(
        i=0.0,
        eta_curr=jnp.array(0.0),
        eta_past=jnp.array(0.0),
        S_curr=jnp.eye(k)/k*trace_ub,
        S_past=jnp.zeros((k,k)),
        max_value_change=jnp.array(1.1*apgd_eps))

    final_apgd_state = bounded_while_loop(
        lambda apgd_state: apgd_state.max_value_change > apgd_eps,
        apgd, 
        init_apgd_state,
        max_steps=apgd_max_iters)

    return final_apgd_state.eta_curr, final_apgd_state.S_curr


def specbm(
    X: Array,
    y: Array,
    z: Array,
    primal_obj_val: float,
    V: Array,
    n: int,
    m: int,
    trace_ub: float,
    C: csc_matrix,
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

    X_bar = jnp.copy(X)
    z_bar = jnp.copy(z)

    bar_obj_val = primal_obj_val

    C_matmat = jax.vmap(C_matvec, 1, 1)
    A_adjoint_batched = jax.vmap(A_adjoint_slim, (None, 1), 1)
    A_operator_batched = jax.vmap(A_operator_slim, 1, 1)

    # TODO: check `solve_subproblem` against SCS and MOSEK

    k = k_curr + k_past

    #S = cp.Variable((k,k), symmetric=True)
    #eta = cp.Variable((1,))
    #constraints = [S >> 0]
    #constraints += [eta >= 0]
    #constraints += [cp.trace(S) + eta*cp.trace(X) <= trace_ub]
    #prob = cp.Problem(
    #    cp.Minimize(y @ b
    #                + cp.trace((eta * X + V @ S @ V.T) @ (C - cp.diag(y)))
    #                + (0.5 / rho) * cp.sum_squares(b - cp.diag(eta * X + V @ S @ V.T))),
    #    constraints)
    #prob.solve(solver=cp.SCS, verbose=True)

    #jax.debug.print("SCS eta: {eta}", eta=eta.value)
    #jax.debug.print("SCS S: {S}", S=S.value)

    eta, S = solve_subproblem(
        C_matmat=C_matmat,
        A_operator_batched=A_operator_batched,
        A_adjoint_batched=A_adjoint_batched,
        b=b,
        trace_ub=trace_ub,
        rho=rho,
        primal_obj_val=primal_obj_val,
        z=z,
        y=y,
        V=V,
        k=k,
        apgd_step_size=apgd_step_size,
        apgd_max_iters=apgd_max_iters,
        apgd_eps=apgd_eps,
    )
    S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
    S_eigvals = jnp.clip(S_eigvals, a_min=0)

    # TODO: fix this for X and X_bar
    VSV_T_factor = (V @ S_eigvecs) * jnp.sqrt(S_eigvals).reshape(1, -1)
    A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)

    X_next = eta * X + V @ S @ V.T
    z_next = eta * z_bar + A_operator_VSV_T
    y_cand = y + (1.0 / rho) * (b - z_next)

    prev_eigvals, prev_eigvecs = approx_grad_k_min_eigen(
        C_matvec=C_matvec,
        A_adjoint_slim=A_adjoint_slim,
        adjoint_left_vec=-y,
        n=n,
        k=k,
        num_iters=lanczos_num_iters,
        rng=jax.random.PRNGKey(0))
    prev_eigvals = -prev_eigvals

    cand_eigvals, cand_eigvecs = approx_grad_k_min_eigen(
        C_matvec=C_matvec,
        A_adjoint_slim=A_adjoint_slim,
        adjoint_left_vec=-y_cand,
        n=n,
        k=k_curr,
        num_iters=lanczos_num_iters,
        rng=jax.random.PRNGKey(0))
    cand_eigvals = -cand_eigvals

    M1 = A_adjoint(y) - C.todense()
    M2 = A_adjoint(y_cand) - C.todense()

    eigvals1, eigvecs1 = jnp.linalg.eigh(M1)
    eigvals2, eigvecs2 = jnp.linalg.eigh(M2)

    prev_dual_obj = jnp.dot(-b, y) + trace_ub*jnp.clip(prev_eigvals[0], a_min=0)
    cand_dual_obj = jnp.dot(-b, y_cand) + trace_ub*jnp.clip(cand_eigvals[0], a_min=0)
    cand_est_obj = jnp.dot(-b, y_cand) + eta*jnp.dot(z_bar, y_cand) - eta*bar_obj_val
    cand_est_obj += jnp.dot(A_operator_VSV_T, y_cand) - jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor))

    y_next = lax.cond(
        beta * (prev_dual_obj - cand_est_obj) <= prev_dual_obj - cand_dual_obj,
        lambda _: y_cand,
        lambda _: y,
        None)

    X_bar_next = eta * X_bar + V @ (
        (S_eigvecs[:, :k_curr] * S_eigvals[:k_curr].reshape(1, -1)) @ S_eigvecs[:, :k_curr].T) @ V.T
    # z_bar_next = 
    V_next = jnp.concatenate([V @ S_eigvecs[:,k_curr:], cand_eigvecs], axis=1)

    embed()
    exit()


    # TODO: update `bar_obj_val`, `z_bar`

    # TODO: implement f and f_bar 
    # TODO: create new Lanczos function for this solver
    # TODO: implement stopping criteria

    # TODO: fix to return all things needed for warm-start
    return jnp.zeros(n, n), jnp.zeros((m,))
