from collections import namedtuple
import cvxpy as cp
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
from functools import partial
import jax
from jax._src.typing import Array
import jax.experimental.host_callback as hcb
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from jax import lax
import time
from typing import Any, Callable, Optional, Tuple, Union

from scipy.sparse import csc_matrix  # type: ignore

from solver.eigen import approx_grad_k_min_eigen

from IPython import embed


def specbm_slow(
    X: Union[Array, None],
    P: Union[Array, None],
    y: Array,
    z: Array,
    primal_obj: float,
    tr_X: float,
    n: int,
    m: int,
    trace_ub: float,
    C: csc_matrix,
    C_matvec: Callable[[Array], Array],
    A_operator_slim: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    Q_base: Callable[[Array], Array],
    U: BCOO,
    Omega: Union[Array, None],
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
    callback_fn: Union[Callable[[Array, Array], Array], None]
) -> Tuple[Array, Array, Array, Array, Array]:

    C_matmat = jax.vmap(C_matvec, 1, 1)
    A_adjoint_batched = jax.vmap(A_adjoint_slim, (None, 1), 1)
    A_operator_batched = jax.vmap(A_operator_slim, 1, 1)

    k = k_curr + k_past

    StateStruct = namedtuple(
        "StateStruct",
        ["t", 
         "X",
         "tr_X",
         "X_bar",
         "tr_X_bar",
         "P",
         "P_bar",
         "z",
         "z_bar",
         "y",
         "V",
         "primal_obj",
         "bar_primal_obj",
         "pen_dual_obj",
         "lb_spec_est"])

    @jax.jit
    def cond_func(state: StateStruct) -> Array:
        return jnp.logical_or(
            state.t == 0, (state.pen_dual_obj - state.lb_spec_est) / (1.0 + state.pen_dual_obj) > eps)

    #@jax.jit
    def body_func(state: StateStruct) -> StateStruct:

        jax.debug.print("start_time: {time}",
                        time=hcb.call(lambda _: time.time(), arg=0, result_shape=float))

        eta, S = solve_quadratic_subproblem(
            C_matmat=C_matmat,
            A_adjoint_batched=A_adjoint_batched,
            Q_base=Q_base,
            U=U,
            b=b,
            trace_ub=trace_ub,
            rho=rho,
            bar_primal_obj=state.bar_primal_obj,
            tr_X_bar=state.tr_X_bar,
            z_bar=state.z_bar,
            y=state.y,
            V=state.V,
            k=k)

        ###################################################################################

        #X = state.X_bar
        #y = state.y
        #V = state.V

        #S_ = cp.Variable((k,k), symmetric=True)
        #eta_ = cp.Variable((1,))
        #constraints = [S_ >> 0]
        #constraints += [eta_ >= 0]
        #constraints += [cp.trace(S_) + eta_*cp.trace(X) <= trace_ub]
        #prob = cp.Problem(
        #    cp.Minimize(y @ b
        #                + cp.trace((eta_ * X + V @ S_ @ V.T) @ (C - cp.diag(y)))
        #                + (0.5 / rho) * cp.sum_squares(b - cp.diag(eta_ * X + V @ S_ @ V.T))),
        #    constraints)
        #prob.solve(solver=cp.SCS, verbose=True)

        ##S = S_.value
        ##eta = eta_.value

        #if state.t == 1:
        #    embed()
        #    exit()

        #del S_
        #del eta_

        #S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
        #S_eigvals = jnp.clip(S_eigvals, a_min=0)    # numerical instability handling

        ################################################################################

        S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
        S_eigvals = jnp.clip(S_eigvals, a_min=0)    # numerical instability handling

        VSV_T_factor = (state.V @ S_eigvecs) * jnp.sqrt(S_eigvals).reshape(1, -1)
        A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)
        if Omega is None:
            X_next = eta * state.X_bar + state.V @ S @ state.V.T
            P_next = None
        else:
            X_next = None
            P_next = eta * state.P_bar + VSV_T_factor @ (VSV_T_factor.T @ Omega)
        tr_X_next = eta * state.tr_X_bar + jnp.trace(S)
        z_next = eta * state.z_bar + A_operator_VSV_T
        y_cand = state.y + (1.0 / rho) * (b - z_next)
        primal_obj_next = eta * state.bar_primal_obj + jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor))

        cand_eigvals, cand_eigvecs = approx_grad_k_min_eigen(
            C_matvec=C_matvec,
            A_adjoint_slim=A_adjoint_slim,
            adjoint_left_vec=-y_cand,
            n=n,
            k=k_curr,
            num_iters=lanczos_num_iters,
            rng=jax.random.PRNGKey(state.t))
        cand_eigvals = -cand_eigvals
        cand_pen_dual_obj = jnp.dot(-b, y_cand) + trace_ub*jnp.clip(cand_eigvals[0], a_min=0)

        # recompute this everytime for safety
        prev_eigvals, _ = approx_grad_k_min_eigen(
            C_matvec=C_matvec,
            A_adjoint_slim=A_adjoint_slim,
            adjoint_left_vec=-state.y,
            n=n,
            k=1,
            num_iters=lanczos_num_iters,
            rng=jax.random.PRNGKey(state.t))
        prev_eigvals = -prev_eigvals
        pen_dual_obj = jnp.dot(-b, state.y) + trace_ub*jnp.clip(prev_eigvals[0], a_min=0)
        pen_dual_obj = jnp.clip(state.pen_dual_obj, a_min=pen_dual_obj)
        #pen_dual_obj = state.pen_dual_obj

        #####################################################################################

        #X = state.X_bar
        #y = y_cand
        #V = state.V

        #S_ = cp.Variable((k,k), symmetric=True)
        #eta_ = cp.Variable((1,))
        #constraints = [S_ >> 0]
        #constraints += [eta_ >= 0]
        #constraints += [cp.trace(S_) + eta_*cp.trace(X) <= trace_ub]
        #prob = cp.Problem(
        #    cp.Maximize(-y @ b + cp.trace((eta_ * X + V @ S_ @ V.T) @ (cp.diag(y) - C.todense()))),
        #    constraints)
        #prob.solve(solver=cp.SCS, verbose=True)

        #_S_eigvals, _S_eigvecs = jnp.linalg.eigh(S_.value)
        #_S_eigvals = jnp.clip(_S_eigvals, a_min=0.0)
        #VSV_T_factor = (V @ (_S_eigvecs)) * jnp.sqrt(_S_eigvals).reshape(1, -1)
        #A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)
        #lb_spec_est_ = (jnp.dot(-b, y) + eta_.value*jnp.dot(y_cand, state.z_bar) + jnp.dot(y_cand, A_operator_VSV_T)
        #            - eta_.value*state.bar_primal_obj - jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor)))
        #lb_spec_est_ = jnp.sum(lb_spec_est_)

        #####################################################################################

        lb_spec_est = compute_lb_spec_est(
            C_matmat=C_matmat,
            A_adjoint_batched=A_adjoint_batched,
            U=U,
            b=b,
            trace_ub=trace_ub,
            bar_primal_obj=state.bar_primal_obj,
            tr_X_bar=state.tr_X_bar,
            z_bar=state.z_bar,
            y=y_cand,
            V=state.V,
            k=k)

        #y_next, pen_dual_obj_next = lax.cond(
        #    beta * (state.pen_dual_obj - lb_spec_est) <= state.pen_dual_obj - cand_pen_dual_obj,
        #    lambda _: (y_cand, cand_pen_dual_obj),
        #    lambda _: (state.y, state.pen_dual_obj),
        #    None)
        y_next, pen_dual_obj_next = lax.cond(
            beta * (pen_dual_obj - lb_spec_est) <= pen_dual_obj - cand_pen_dual_obj,
            lambda _: (y_cand, cand_pen_dual_obj),
            lambda _: (state.y, pen_dual_obj),
            None)

        curr_VSV_T_factor = (state.V @ S_eigvecs[:, :k_curr]) * jnp.sqrt(S_eigvals[:k_curr]).reshape(1, -1)
        if Omega is None:
            X_bar_next = eta * state.X_bar + curr_VSV_T_factor @ curr_VSV_T_factor.T
            P_bar_next = None
        else:
            X_bar_next = None
            P_bar_next = eta * state.P_bar + curr_VSV_T_factor @ (curr_VSV_T_factor.T @ Omega)
        tr_X_bar_next = eta * state.tr_X_bar + jnp.sum(S_eigvals[:k_curr])
        z_bar_next =  eta * state.z_bar + jnp.sum(A_operator_batched(curr_VSV_T_factor), axis=1)
        V_next = jnp.concatenate([state.V @ S_eigvecs[:,k_curr:], cand_eigvecs], axis=1)
        bar_primal_obj_next = eta * state.bar_primal_obj
        bar_primal_obj_next += jnp.trace(curr_VSV_T_factor.T @ C_matmat(curr_VSV_T_factor))
        
        obj_val = primal_obj_next / (SCALE_C * SCALE_X)
        infeas_gap = jnp.linalg.norm((state.z - b) / SCALE_X) 
        infeas_gap /= 1.0 + jnp.linalg.norm(b / SCALE_X)
        max_infeas = jnp.max(jnp.abs(state.z - b)) / SCALE_X

        if Omega is not None and callback_fn is not None:
            callback_val = callback_fn(Omega, state.P)
        else:
            callback_val = None

        end_time = hcb.call(lambda _: time.time(), arg=0, result_shape=float)
        jax.debug.print("t: {t} - end_time: {end_time} - pen_dual_obj: {pen_dual_obj}"
                        " - cand_pen_dual_obj: {cand_pen_dual_obj} - lb_spec_est: {lb_spec_est}"
                        " - pen_dual_obj_next: {pen_dual_obj_next} - primal_obj: {primal_obj}"
                        " - obj_gap: {obj_gap} - infeas_gap: {infeas_gap}"
                        " - max_infeas: {max_infeas} - callback_val: {callback_val}",
                        t=state.t,
                        end_time=end_time,
                        pen_dual_obj=state.pen_dual_obj,
                        cand_pen_dual_obj=cand_pen_dual_obj,
                        lb_spec_est=lb_spec_est,
                        pen_dual_obj_next=pen_dual_obj_next,
                        primal_obj=primal_obj_next,
                        obj_gap=jnp.abs(obj_val + pen_dual_obj_next) / (1.0 + jnp.abs(obj_val)),
                        infeas_gap=infeas_gap,
                        max_infeas=max_infeas,
                        callback_val=callback_val)

        return StateStruct(
            t=state.t+1,
            X=X_next,
            tr_X=tr_X_next,
            X_bar=X_bar_next,
            tr_X_bar=tr_X_bar_next,
            P=P_next,
            P_bar=P_bar_next,
            z=z_next,
            z_bar=z_bar_next,
            y=y_next,
            V=V_next,
            primal_obj=primal_obj_next,
            bar_primal_obj=bar_primal_obj_next,
            pen_dual_obj=pen_dual_obj_next,
            lb_spec_est=lb_spec_est)

    # compute current `pen_dual_obj` for `init_state`
    prev_eigvals, prev_eigvecs = approx_grad_k_min_eigen(
        C_matvec=C_matvec,
        A_adjoint_slim=A_adjoint_slim,
        adjoint_left_vec=-y,
        n=n,
        k=k,
        num_iters=lanczos_num_iters,
        rng=jax.random.PRNGKey(0))
    
    prev_eigvals = -prev_eigvals
    pen_dual_obj = jnp.dot(-b, y) + trace_ub*jnp.clip(prev_eigvals[0], a_min=0)

    init_state = StateStruct(
        t=0,
        X=X,
        tr_X=tr_X,
        X_bar=X,
        tr_X_bar=tr_X,
        P=P,
        P_bar=P,
        z=z,
        z_bar=z,
        y=y,
        V=prev_eigvecs,
        primal_obj=primal_obj,
        bar_primal_obj=primal_obj,
        pen_dual_obj=pen_dual_obj,
        lb_spec_est=0.0)

    state = init_state
    for _ in range(80):
        state = body_func(state)

    embed()
    exit()

    import pickle
    with open("state.pkl", "rb") as f:
        state = StateStruct(*pickle.load(f))

    state = body_func(state)

    embed()
    exit()

    final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=max_iters)

    embed()
    exit()

    return (final_state.X,
            final_state.P,
            final_state.y,
            final_state.z,
            final_state.primal_obj,
            final_state.tr_X)