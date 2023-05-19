from collections import namedtuple
import cvxpy as cp
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
from functools import partial
import jax
from jax._src.abstract_arrays import ShapedArray
from jax._src.typing import Array
import jax.experimental.host_callback as hcb
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from jax import lax
import numpy as np
import time
from typing import Any, Callable, Optional, Tuple, Union

from scipy.sparse import csc_matrix  # type: ignore

from solver.eigen import approx_grad_k_min_eigen
from solver.utils import apply_A_operator_batched, apply_A_adjoint_batched

from IPython import embed


def create_solve_quad_subprob(
    n: int,
    m: int,
    k: int,
    C: Array,
    A_tensor: Array,
    b: Array,
    trace_ub: float,
    rho: float
) -> Callable[[Array, Array, Array], Tuple[Array, Array]]:
    create_solve_quad_subprob.X_bar = cp.Parameter((n, n))
    create_solve_quad_subprob.A_adj_y = cp.Parameter((n, n))
    create_solve_quad_subprob.V = cp.Parameter((n, k))

    A_matrices = list(A_tensor)

    create_solve_quad_subprob.S = cp.Variable((k,k), symmetric=True)
    create_solve_quad_subprob.eta = cp.Variable((1,))
    constraints = [create_solve_quad_subprob.S >> 0]
    constraints += [create_solve_quad_subprob.eta >= 0]
    constraints += [cp.trace(create_solve_quad_subprob.S)
                    + create_solve_quad_subprob.eta*cp.trace(create_solve_quad_subprob.X_bar)
                    <= trace_ub]
    create_solve_quad_subprob.prob = cp.Problem(
        cp.Minimize(cp.trace((create_solve_quad_subprob.eta * create_solve_quad_subprob.X_bar
                              + create_solve_quad_subprob.V 
                                @ create_solve_quad_subprob.S 
                                @ create_solve_quad_subprob.V.T) 
                              @ (C - create_solve_quad_subprob.A_adj_y))
                    + (0.5 / rho) * cp.sum_squares(
                        cp.hstack([b[i] - cp.trace(A_i @ (create_solve_quad_subprob.eta
                                                * create_solve_quad_subprob.X_bar 
                                             + create_solve_quad_subprob.V
                                                @ create_solve_quad_subprob.S
                                                @ create_solve_quad_subprob.V.T))
                             for i, A_i in enumerate(A_matrices)]))),
        constraints)
    
    def solve_quad_subprob(X_bar: Array, y: Array, V: Array) -> Tuple[Array, Array]:
        create_solve_quad_subprob.X_bar.value = np.asarray(X_bar)
        create_solve_quad_subprob.A_adj_y.value = np.asarray(jnp.sum(A_tensor * y.reshape(m, 1, 1), axis=0))
        create_solve_quad_subprob.V.value = np.asarray(V)
        create_solve_quad_subprob.prob.solve(solver=cp.SCS, verbose=False)
        return (jnp.array(create_solve_quad_subprob.eta.value),
                jnp.array(create_solve_quad_subprob.S.value))

    return solve_quad_subprob


def create_lb_spec_est(
    n: int,
    m: int,
    k: int,
    C: Array,
    A_tensor: Array,
    b: Array,
    trace_ub: float,
) -> Callable[[Array, Array, Array], Array]:
    create_lb_spec_est.X_bar = cp.Parameter((n, n))
    create_lb_spec_est.A_adj_y = cp.Parameter((n, n))
    create_lb_spec_est.V = cp.Parameter((n, k))

    create_lb_spec_est.S = cp.Variable((k,k), symmetric=True)
    create_lb_spec_est.eta = cp.Variable((1,))
    constraints = [create_lb_spec_est.S >> 0]
    constraints += [create_lb_spec_est.eta >= 0]
    constraints += [cp.trace(create_lb_spec_est.S)
                    + create_lb_spec_est.eta*cp.trace(create_lb_spec_est.X_bar)
                    <= trace_ub]
    create_lb_spec_est.prob = cp.Problem(
        cp.Minimize(cp.trace((create_lb_spec_est.eta * create_lb_spec_est.X_bar
                              + create_lb_spec_est.V 
                                @ create_lb_spec_est.S 
                                @ create_lb_spec_est.V.T) 
                              @ (C - create_lb_spec_est.A_adj_y))),
        constraints)
    
    def compute_lb_spec_est(X_bar: Array, y: Array, V: Array) -> Array:
        create_lb_spec_est.X_bar.value = np.asarray(X_bar)
        create_lb_spec_est.A_adj_y.value = np.asarray(jnp.sum(A_tensor * y.reshape(m, 1, 1), axis=0))
        create_lb_spec_est.V.value = np.asarray(V)
        create_lb_spec_est.prob.solve(solver=cp.SCS, verbose=False)
        lb_spec_est = jnp.dot(b, y)
        lb_spec_est += jnp.trace(jnp.matmul(
            create_lb_spec_est.eta.value * X_bar + V @ create_lb_spec_est.S.value @ V.T,
            C - create_lb_spec_est.A_adj_y.value))
        return -lb_spec_est

    return compute_lb_spec_est


def specbm_slow(
    X: Union[Array, None],
    y: Array,
    z: Array,
    primal_obj: float,
    tr_X: float,
    n: int,
    m: int,
    trace_ub: float,
    C: BCOO,
    C_dense: Array,
    A_data: Array,
    A_indices: Array,
    A_tensor: Array,
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

    k = k_curr + k_past
    solve_quad_subprob = create_solve_quad_subprob(n, m, k, C_dense, A_tensor, b, trace_ub, rho)
    compute_lb_spec_est = create_lb_spec_est(n, m, k, C_dense, A_tensor, b, trace_ub)

    StateStruct = namedtuple(
        "StateStruct",
        ["t", 
         "X",
         "tr_X",
         "X_bar",
         "tr_X_bar",
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

    @jax.jit
    def body_func(state: StateStruct) -> StateStruct:

        jax.debug.print("start_time: {time}",
                        time=hcb.call(lambda _: time.time(), arg=0, result_shape=float))

        eta, S = hcb.call(lambda arg: solve_quad_subprob(*arg),
                          arg=(state.X_bar, state.y, state.V),
                          result_shape=((ShapedArray([1], float), ShapedArray([k, k], float))))

        S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
        S_eigvals = jnp.clip(S_eigvals, a_min=0)    # numerical instability handling
        VSV_T = state.V @ S @ state.V.T
        X_next = eta * state.X_bar + VSV_T
        tr_X_next = (eta * state.tr_X_bar + jnp.trace(S)).squeeze()
        z_next = eta * state.z_bar + jnp.sum(
            jnp.sum(A_tensor * VSV_T.reshape((1, n, n)), axis=2),
            axis=1)
        y_cand = state.y + (1.0 / rho) * (b - z_next)
        primal_obj_next = eta * state.bar_primal_obj + jnp.trace(C @ VSV_T)
        primal_obj_next = primal_obj_next.squeeze()

        cand_eigvals, cand_eigvecs = jnp.linalg.eigh(
            C - jnp.sum(A_tensor * y_cand.reshape(m, 1, 1), axis=0))
        cand_eigvals = cand_eigvals[:k_curr]
        cand_eigvecs = cand_eigvecs[:, :k_curr]
        cand_eigvals = -cand_eigvals
        cand_pen_dual_obj = jnp.dot(-b, y_cand) + trace_ub*jnp.clip(cand_eigvals[0], a_min=0)

        lb_spec_est = hcb.call(lambda arg: compute_lb_spec_est(*arg),
                               arg=(state.X_bar, y_cand, state.V),
                               result_shape=float)

        y_next, pen_dual_obj_next = lax.cond(
            beta * (state.pen_dual_obj - lb_spec_est) <= state.pen_dual_obj - cand_pen_dual_obj,
            lambda _: (y_cand, cand_pen_dual_obj),
            lambda _: (state.y, state.pen_dual_obj),
            None)

        # TODO: fix use of `k_past` with QR factorization
        curr_VSV_T_factor = (state.V @ S_eigvecs[:, :k_curr]) * jnp.sqrt(S_eigvals[:k_curr]).reshape(1, -1)
        X_bar_next = eta * state.X_bar + curr_VSV_T_factor @ curr_VSV_T_factor.T
        tr_X_bar_next = (eta * state.tr_X_bar + jnp.sum(S_eigvals[:k_curr])).squeeze()
        z_bar_next = eta * state.z_bar + jnp.sum(
            jnp.sum(A_tensor * (curr_VSV_T_factor @ curr_VSV_T_factor.T).reshape((1, n, n)), axis=2),
            axis=1)
        V_next, _ = jnp.linalg.qr(jnp.concatenate([state.V @ S_eigvecs[:,k_curr:], cand_eigvecs], axis=1))
        bar_primal_obj_next = jnp.trace(C @ X_bar_next)
        
        obj_val = primal_obj_next / (SCALE_C * SCALE_X)
        infeas_gap = jnp.linalg.norm((state.z - b) / SCALE_X) 
        infeas_gap /= 1.0 + jnp.linalg.norm(b / SCALE_X)
        max_infeas = jnp.max(jnp.abs(state.z - b)) / SCALE_X

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
            z=z_next,
            z_bar=z_bar_next,
            y=y_next,
            V=V_next,
            primal_obj=primal_obj_next,
            bar_primal_obj=bar_primal_obj_next,
            pen_dual_obj=pen_dual_obj_next,
            lb_spec_est=lb_spec_est)

    init_eigvals_slow, init_eigvecs_slow = jnp.linalg.eigh(
        C_dense - jnp.sum(A_tensor * y.reshape(m, 1, 1), axis=0))
    init_eigvals_slow = init_eigvals_slow[:k]
    init_eigvecs_slow = init_eigvecs_slow[:, :k]
    init_eigvals_slow = -init_eigvals_slow

    init_eigvals, init_eigvecs = approx_grad_k_min_eigen(
        C=C,
        A_data=A_data,
        A_indices=A_indices,
        adjoint_left_vec=y,
        n=n,
        k=k,
        num_iters=lanczos_num_iters,
        rng=jax.random.PRNGKey(-1))

    embed()
    exit()

    init_pen_dual_obj = jnp.dot(-b, y) + trace_ub*jnp.clip(init_eigvals[0], a_min=0)

    init_state = StateStruct(
        t=jnp.array(0),
        X=X,
        tr_X=tr_X,
        X_bar=X,
        tr_X_bar=tr_X,
        z=z,
        z_bar=z,
        y=y,
        V=init_eigvecs,
        primal_obj=primal_obj,
        bar_primal_obj=primal_obj,
        pen_dual_obj=init_pen_dual_obj,
        lb_spec_est=jnp.array(0.0))

    final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=1000)

    embed()
    exit()

    state = init_state
    for _ in range(5):
        state = body_func(state)

    embed()
    exit()



    import pickle
    with open("state.pkl", "rb") as f:
        state = StateStruct(*pickle.load(f))

    state = body_func(state)

    embed()
    exit()


    return (final_state.X,
            final_state.y,
            final_state.z,
            final_state.primal_obj,
            final_state.tr_X)