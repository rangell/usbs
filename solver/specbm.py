from collections import namedtuple
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
from typing import Callable, Tuple, Union, Any

from solver.lanczos import eigsh_smallest
from solver.utils import (apply_A_operator_batched,
                          apply_A_adjoint_batched,
                          create_svec_matrix,
                          create_Q_base)
from utils.common import SDPState

from IPython import embed


@partial(jax.jit, static_argnames=["n", "m", "k", "ipm_eps", "ipm_max_iters"])
def solve_quad_subprob_ipm(
    C: BCOO,
    A_data: Array,
    A_indices: Array,
    U: BCOO,
    b: Array,
    trace_ub: float,
    rho: float,
    bar_primal_obj: Array,
    z_bar: Array,
    tr_X_bar: Array,
    y: Array,
    V: Array,
    n: int,
    m: int,
    k: int,
    eta_init: Array,
    S_init: Array,
    ipm_eps: float,
    ipm_max_iters: int
) -> Tuple[Array, Array]:

    svec = lambda mx: U @ mx.reshape(-1)
    svec_inv = lambda vec: (U.T @ vec).reshape(k, k)

    # create problem constants
    tr_X_bar = lax.cond(tr_X_bar > 0.0, lambda _: tr_X_bar, lambda _: trace_ub, None)
    Q_11 = (trace_ub**2 / rho) * create_Q_base(m, k, U, A_data, A_indices, V)
    q_12 = (trace_ub**2 / (rho * tr_X_bar)) * svec(
        V.T @ apply_A_adjoint_batched(n, A_data, A_indices, z_bar, V))
    q_22 = (trace_ub**2 / (rho * tr_X_bar**2)) * jnp.dot(z_bar, z_bar)
    h_1 = trace_ub * svec(V.T @ (C @ V
                                 - apply_A_adjoint_batched(n, A_data, A_indices, y, V)
                                 - apply_A_adjoint_batched(n, A_data, A_indices, b, V) / rho))
    h_2 = (trace_ub / tr_X_bar) * (bar_primal_obj - jnp.dot(z_bar, y) - (jnp.dot(z_bar, b) / rho))
    svec_I = svec(jnp.eye(k))

    eta_init = jnp.asarray([(tr_X_bar / trace_ub) * eta_init])
    S_init = S_init / trace_ub

    T_init = svec_inv(Q_11 @ svec(S_init) + eta_init * q_12 + h_1)
    zeta_init = jnp.dot(q_12, svec(S_init)) + eta_init * q_22 + h_2
    dual_infeas_vals = jnp.append(jnp.diag(T_init), zeta_init)
    dual_infeas_width = jnp.clip(jnp.max(dual_infeas_vals) - jnp.min(dual_infeas_vals), a_min=100.0)
    omega_init = jnp.asarray([-1.01 * jnp.min(dual_infeas_vals)])
    omega_init = jnp.clip(omega_init, a_min=dual_infeas_width)
    T_init += omega_init * jnp.eye(k)
    zeta_init += omega_init
    mu_init = (jnp.dot(svec(S_init), svec(T_init))
               + eta_init * zeta_init
               + omega_init * (1 - jnp.dot(svec_I, svec(S_init)) - eta_init)) / (2*k + 4.0)

    IPMState = namedtuple("IPMState", ["i", "S", "eta", "T", "zeta", "omega", "mu"])

    @jax.jit
    def body_func(ipm_state: IPMState) -> IPMState:
        kappa_1 = (1 - jnp.dot(svec_I, svec(ipm_state.S)) - ipm_state.eta) / ipm_state.omega
        kappa_2 = (ipm_state.zeta / ipm_state.eta) + q_22
        F_1 = Q_11 @ svec(ipm_state.S) + ipm_state.eta * q_12 + h_1
        F_1 += -svec(ipm_state.T) + ipm_state.omega * svec_I
        F_2 = jnp.dot(q_12, svec(ipm_state.S)) + ipm_state.eta * q_22 + h_2
        F_2 += -ipm_state.zeta + ipm_state.omega

        # create and solve linear system for S update direction
        S_inv = jnp.linalg.pinv(ipm_state.S)
        T_sym_kron_S_inv = 0.5 * U @ (jnp.kron(ipm_state.T, S_inv)
                                      + jnp.kron(S_inv, ipm_state.T)) @ U.T
        coeff_mx = Q_11 + T_sym_kron_S_inv
        coeff_mx -= (jnp.outer(q_12, kappa_1 * q_12 + svec_I)
                     + jnp.outer(svec_I, q_12 - kappa_2 * svec_I)) / (kappa_1 * kappa_2 + 1)
        ordinate_vals = -F_1 + ipm_state.mu * svec(S_inv) - svec(ipm_state.T)
        ordinate_vals += ((ipm_state.mu / ipm_state.omega
                           + jnp.dot(svec_I, svec(ipm_state.S))
                           + ipm_state.eta - 1
                           + kappa_1 * (F_2 - ipm_state.mu / ipm_state.eta + ipm_state.zeta))
                          / (kappa_1 * kappa_2 + 1)) * q_12
        ordinate_vals += ((F_2 - ipm_state.mu / ipm_state.eta + ipm_state.zeta
                           - kappa_2*(ipm_state.mu / ipm_state.omega
                                      + jnp.dot(svec_I, svec(ipm_state.S))
                                      + ipm_state.eta - 1))
                          / (kappa_1 * kappa_2 + 1)) * svec_I
        delta_svec_S = jnp.linalg.solve(coeff_mx, ordinate_vals)

        # substitute back in to get all of the other update directions
        delta_eta = (-jnp.dot(kappa_1 * q_12 + svec_I, delta_svec_S)
                     - (ipm_state.mu / ipm_state.omega)
                     - jnp.dot(svec_I, svec(ipm_state.S))
                     - ipm_state.eta + 1
                     + kappa_1 * (-F_2 + ipm_state.mu / ipm_state.eta - ipm_state.zeta)
                    ) / (kappa_1 * kappa_2 + 1)
        delta_omega = (-F_2 + ipm_state.mu / ipm_state.eta - ipm_state.zeta
                       + kappa_2 * (ipm_state.mu / ipm_state.omega
                                    + jnp.dot(svec_I, svec(ipm_state.S)) + ipm_state.eta - 1)
                       - jnp.dot(q_12 - kappa_2 * svec_I, delta_svec_S)) / (kappa_1 * kappa_2 + 1)
        delta_svec_T = Q_11 @ delta_svec_S + delta_eta * q_12 + delta_omega * svec_I + F_1
        delta_zeta = jnp.dot(q_12, delta_svec_S) + delta_eta * q_22 + delta_omega + F_2

        # compute step size
        delta_S = svec_inv(delta_svec_S)
        delta_T = svec_inv(delta_svec_T)

        step_size_numers = jnp.concatenate([ipm_state.eta, ipm_state.zeta, ipm_state.omega])
        step_size_denoms = jnp.concatenate([delta_eta, delta_zeta, delta_omega])

        step_size = jnp.clip(-1.0 / jnp.min(step_size_denoms / step_size_numers), a_max=1.0)
        step_size = 0.99 * lax.cond(step_size <= 0.0, lambda _: 1.0, lambda _: step_size, None)
        step_size = bounded_while_loop(
            lambda step_size: jnp.linalg.eigh(ipm_state.S + step_size * delta_S)[0][0] < 0.0,
            lambda step_size: step_size * 0.8, 
            step_size,
            max_steps=100)
        step_size = bounded_while_loop(
            lambda step_size: jnp.linalg.eigh(ipm_state.T + step_size * delta_T)[0][0] < 0.0,
            lambda step_size: step_size * 0.8, 
            step_size,
            max_steps=100)

        S_next = ipm_state.S + step_size * delta_S
        eta_next = ipm_state.eta + step_size * delta_eta
        T_next = ipm_state.T + step_size * delta_T
        zeta_next = ipm_state.zeta + step_size * delta_zeta
        omega_next = ipm_state.omega + step_size * delta_omega

        mu_next = (jnp.dot(svec(S_next), svec(T_next))
                   + eta_next * zeta_next
                   + omega_next * (1 - jnp.dot(svec_I, svec(S_next)) - eta_next)) / (2*k + 4.0)
        mu_next *= lax.cond(step_size > 0.2, lambda _: 0.5 - 0.4 * step_size**2, lambda _: 1.0, None)
        mu_next = jnp.clip(mu_next, a_max=ipm_state.mu)
        return IPMState(
            i=ipm_state.i+1,
            S=S_next,
            eta=eta_next,
            T=T_next,
            zeta=zeta_next,
            omega=omega_next,
            mu=mu_next)

    init_ipm_state = IPMState(
        i=0, S=S_init, eta=eta_init, T=T_init, zeta=zeta_init, omega=omega_init, mu=mu_init)
    
    final_ipm_state = bounded_while_loop(
        lambda ipm_state: ipm_state.mu.squeeze() > ipm_eps,
        body_func, 
        init_ipm_state,
        max_steps=ipm_max_iters)

    return ((trace_ub / tr_X_bar) * final_ipm_state.eta.squeeze(),
            trace_ub * final_ipm_state.S)


@partial(jax.jit, static_argnames=["n", "k", "ipm_eps", "ipm_max_iters"])
def compute_lb_spec_est_ipm(
    C: BCOO,
    A_data: Array,
    A_indices: Array,
    U: BCOO,
    b: Array,
    trace_ub: float,
    bar_primal_obj: Array,
    z_bar: Array,
    tr_X_bar: Array,
    y: Array,
    V: Array,
    n: int,
    k: int,
    ipm_eps: float,
    ipm_max_iters: int
) -> Tuple[Array, Array]:

    svec = lambda mx: U @ mx.reshape(-1)
    svec_inv = lambda vec: (U.T @ vec).reshape(k, k)

    # create problem constants
    tr_X_bar = lax.cond(tr_X_bar > 0.0, lambda _: tr_X_bar, lambda _: trace_ub, None)
    g_1 = trace_ub * svec(V.T @ (C @ V)
                          - V.T @ apply_A_adjoint_batched(n, A_data, A_indices, y, V))
    g_2 = (trace_ub / tr_X_bar) * (bar_primal_obj - jnp.dot(z_bar, y))
    svec_I = svec(jnp.eye(k))

    # initialize all Lagrangian variables
    S_init = 0.99 * (jnp.eye(k) / (k + 1.0))
    eta_init = lax.cond(
        bar_primal_obj == 0.0,
        lambda _: jnp.asarray([0.00001]),
        lambda _: jnp.asarray([0.99 * (1.0 / (k + 1.0))]),
        None)

    T_init = svec_inv(g_1)
    zeta_init = g_2
    dual_infeas_vals = jnp.append(jnp.diag(T_init), zeta_init)
    dual_infeas_width = jnp.clip(jnp.max(dual_infeas_vals) - jnp.min(dual_infeas_vals), a_min=100.0)
    omega_init = jnp.asarray([-1.01 * jnp.min(dual_infeas_vals)])
    omega_init = jnp.clip(omega_init, a_min=dual_infeas_width)
    T_init += omega_init * jnp.eye(k)
    zeta_init += omega_init
    mu_init = (jnp.dot(svec(S_init), svec(T_init))
                + eta_init * zeta_init
                + omega_init * (1 - jnp.dot(svec_I, svec(S_init)) - eta_init)) / (2*k + 4.0)

    IPMState = namedtuple("IPMState", ["i", "S", "eta", "T", "zeta", "omega", "mu", "obj_gap"])

    @jax.jit
    def body_func(ipm_state: IPMState) -> IPMState:
        kappa_1 = (1 - jnp.dot(svec_I, svec(ipm_state.S)) - ipm_state.eta) / ipm_state.omega

        # create and solve linear system for S update direction
        S_inv = jnp.linalg.inv(ipm_state.S)
        T_sym_kron_S_inv = 0.5 * U @ (jnp.kron(ipm_state.T, S_inv)
                                      + jnp.kron(S_inv, ipm_state.T)) @ U.T
        coeff_mx = T_sym_kron_S_inv
        coeff_mx += ((ipm_state.zeta / ipm_state.eta)
                     / (kappa_1 * ipm_state.zeta / ipm_state.eta + 1)) * jnp.outer(svec_I, svec_I)
        ordinate_vals = ipm_state.zeta / ipm_state.eta * (
            -kappa_1 * (ipm_state.mu / ipm_state.eta - g_2 - ipm_state.omega)
            + ipm_state.mu / ipm_state.omega
            + jnp.dot(svec_I, svec(ipm_state.S))
            + ipm_state.eta - 1) / (-kappa_1 * ipm_state.zeta / ipm_state.eta - 1) * svec_I
        ordinate_vals += (g_2 - ipm_state.mu / ipm_state.eta) * svec_I - g_1
        ordinate_vals += ipm_state.mu * svec(S_inv)
        delta_svec_S = jnp.linalg.solve(coeff_mx, ordinate_vals)

        ## substitute back in to get all of the other update directions
        delta_eta = (jnp.dot(svec_I, delta_svec_S) 
                     -kappa_1 * (ipm_state.mu / ipm_state.eta - g_2 - ipm_state.omega)
                     + ipm_state.mu / ipm_state.omega
                     + jnp.dot(svec_I, svec(ipm_state.S))
                     + ipm_state.eta - 1) / (-kappa_1 * ipm_state.zeta / ipm_state.eta - 1)
        delta_omega = -ipm_state.zeta / ipm_state.eta * delta_eta + ipm_state.mu / ipm_state.eta
        delta_omega += -g_2 - ipm_state.omega
        delta_svec_T = delta_omega * svec_I - svec(ipm_state.T) + g_1 + ipm_state.omega * svec_I
        delta_zeta = delta_omega - ipm_state.zeta + g_2 + ipm_state.omega

        # compute step size
        delta_S = svec_inv(delta_svec_S)
        delta_T = svec_inv(delta_svec_T)

        step_size_numers = jnp.concatenate([ipm_state.eta, ipm_state.zeta, ipm_state.omega])
        step_size_denoms = jnp.concatenate([delta_eta, delta_zeta, delta_omega])

        step_size = jnp.clip(-1.0 / jnp.min(step_size_denoms / step_size_numers), a_max=1.0)
        step_size = 0.99 * lax.cond(step_size <= 0.0, lambda _: 1.0, lambda _: step_size, None)

        step_size = bounded_while_loop(
            lambda step_size: jnp.linalg.eigh(ipm_state.S + step_size * delta_S)[0][0] < 0.0,
            lambda step_size: step_size * 0.8, 
            step_size,
            max_steps=100)

        step_size = bounded_while_loop(
            lambda step_size: jnp.linalg.eigh(ipm_state.T + step_size * delta_T)[0][0] < 0.0,
            lambda step_size: step_size * 0.8, 
            step_size,
            max_steps=100)

        S_next = ipm_state.S + step_size * delta_S
        eta_next = ipm_state.eta + step_size * delta_eta
        T_next = ipm_state.T + step_size * delta_T
        zeta_next = ipm_state.zeta + step_size * delta_zeta
        omega_next = ipm_state.omega + step_size * delta_omega

        mu_next = (jnp.dot(svec(S_next), svec(T_next))
                   + eta_next * zeta_next
                   + omega_next * (1 - jnp.dot(svec_I, svec(S_next)) - eta_next)) / (2*k + 4.0)
        mu_next *= lax.cond(step_size > 0.2, lambda _: 0.5 - 0.4 * step_size**2, lambda _: 1.0, None)
        mu_next = jnp.clip(mu_next, a_max=ipm_state.mu)

        lb_spec_est = jnp.dot(b, y) + jnp.dot(g_1, svec(ipm_state.S))
        lb_spec_est += ipm_state.eta.squeeze() * g_2
        lb_spec_est = -lb_spec_est.squeeze()

        lb_spec_est_next = jnp.dot(b, y) + jnp.dot(g_1, svec(S_next))
        lb_spec_est_next += eta_next.squeeze() * g_2
        lb_spec_est_next = -lb_spec_est_next.squeeze()

        return IPMState(
            i=ipm_state.i+1,
            S=S_next,
            eta=eta_next,
            T=T_next,
            zeta=zeta_next,
            omega=omega_next,
            mu=mu_next,
            obj_gap=jnp.abs((lb_spec_est - lb_spec_est_next) / lb_spec_est_next))

    init_ipm_state = IPMState(
        i=0,
        S=S_init,
        eta=eta_init,
        T=T_init,
        zeta=zeta_init,
        omega=omega_init,
        mu=mu_init,
        obj_gap=1.0)

    final_ipm_state = bounded_while_loop(
        lambda ipm_state: ipm_state.mu.squeeze() > ipm_eps,
        body_func, 
        init_ipm_state,
        max_steps=ipm_max_iters)

    lb_spec_est = jnp.dot(b, y) + jnp.dot(g_1, svec(final_ipm_state.S))
    lb_spec_est += final_ipm_state.eta.squeeze() * g_2
    return -lb_spec_est.squeeze()


@partial(jax.jit, static_argnames=["n", "m", "k", "subprob_eps", "subprob_max_iters"])
def solve_step_subprob(
    C: BCOO,
    A_data: Array,
    A_indices: Array,
    U: BCOO,
    b: Array,
    b_ineq_mask: Array,
    upsilon: Array,
    trace_ub: float,
    rho: float,
    bar_primal_obj: Array,
    z_bar: Array,
    tr_X_bar: Array,
    y: Array,
    V: Array,
    n: int,
    m: int,
    k: int,
    subprob_eps: float,
    subprob_max_iters: int
) -> Tuple[Array, Array, Array]:

    SubprobStateStruct = namedtuple("SubprobStateStruct", ["i", "eta", "S", "upsilon", "upsilon_gap"])

    def cond_func(state: SubprobStateStruct) -> bool:
        return state.upsilon_gap > subprob_eps

    def body_func(state: SubprobStateStruct) -> SubprobStateStruct:
        eta_next, S_next = solve_quad_subprob_ipm(
            C=C,
            A_data=A_data,
            A_indices=A_indices,
            U=U,
            b=b-state.upsilon,
            trace_ub=trace_ub,
            rho=rho,
            bar_primal_obj=bar_primal_obj,
            z_bar=z_bar,
            tr_X_bar=tr_X_bar,
            y=y,
            V=V,
            n=n,
            m=m,
            k=k,
            eta_init=state.eta,
            S_init=state.S,
            ipm_eps=subprob_eps,
            ipm_max_iters=subprob_max_iters)
        S_eigvals, S_eigvecs = jnp.linalg.eigh(S_next)
        S_eigvals = jnp.clip(S_eigvals, a_min=0)    # numerical instability handling
        VSV_T_factor = (V @ S_eigvecs) * jnp.sqrt(S_eigvals).reshape(1, -1)
        A_operator_VSV_T = apply_A_operator_batched(m, A_data, A_indices, VSV_T_factor)
        z_next = eta_next * z_bar + A_operator_VSV_T
        upsilon_next = b_ineq_mask * jnp.clip(b - z_next + (rho * y), a_min=0.0)
        upsilon_gap = jnp.max(jnp.abs(state.upsilon - upsilon_next))
        _subprob_state = SubprobStateStruct(
            i=state.i + 1, eta=eta_next, S=S_next, upsilon=upsilon_next, upsilon_gap=upsilon_gap)
        return _subprob_state

    
    tr_X_bar = lax.cond(tr_X_bar > 0.0, lambda _: tr_X_bar, lambda _: trace_ub, None)
    S_init = 0.99 * (jnp.eye(k) / (k + 1.0))
    eta_init = lax.cond(
        bar_primal_obj == 0.0,
        lambda _: jnp.asarray([0.00001]),
        lambda _: jnp.asarray([0.99 * (1.0 / (k + 1.0))]),
        None)

    eta_init = (trace_ub / tr_X_bar) * eta_init.squeeze()
    S_init = trace_ub * S_init

    init_state = SubprobStateStruct(
        i=0, eta=eta_init, S=S_init, upsilon=upsilon, upsilon_gap=jnp.array(1.0))
    final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=subprob_max_iters)
    return final_state.eta, final_state.S, final_state.upsilon


def specbm(
    sdp_state: SDPState,
    n: int,
    m: int,
    trace_ub: float,
    trace_factor: float,
    rho: float,
    beta: float,
    k_curr: int,
    k_past: int,
    max_iters: int,
    max_time: float,
    obj_gap_eps: float,
    infeas_gap_eps: float,
    max_infeas_eps: float,
    lanczos_inner_iterations: int,
    lanczos_max_restarts: int,
    subprob_eps: float,
    subprob_max_iters: int,
    callback_fn: Union[Callable[[Array, Array], Array], None],
    callback_static_args: bytes,
    callback_nonstatic_args: Any
) -> Tuple[Array, Array, Array, Array, Array]:

    k = k_curr + k_past
    U = create_svec_matrix(k)
    SCALE_C = sdp_state.SCALE_C 
    SCALE_X = sdp_state.SCALE_X
    SCALE_A = sdp_state.SCALE_A

    StateStruct = namedtuple(
        "StateStruct",
        ["t", 
         "C",
         "A_data",
         "A_indices",
         "b",
         "b_ineq_mask",
         "upsilon",
         "Omega",
         "U",
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
         "q0",
         "callback_nonstatic_args",
         "start_time",
         "curr_time",
         "obj_gap",
         "infeas_gap",
         "max_infeas",
         "primal_obj",
         "bar_primal_obj",
         "pen_dual_obj",
         "lb_spec_est",
         "neg_obj_lb",
         "y_changed"])

    @jax.jit
    def cond_func(state: StateStruct) -> Array:
        # NOTE: bounded_while_loop takes care of max_iters
        return jnp.logical_or(
            jnp.logical_not(state.y_changed),
            jnp.logical_or(
                state.t == 0,
                jnp.logical_and(
                    state.curr_time - state.start_time < max_time,
                    jnp.logical_or(
                        state.obj_gap > obj_gap_eps,
                        jnp.logical_or(state.infeas_gap > infeas_gap_eps,
                                    state.max_infeas > max_infeas_eps)))))

    @jax.jit
    def body_func(state: StateStruct) -> StateStruct:

        jax.debug.print("start_time: {time}",
                        time=hcb.call(lambda _: time.time(), arg=0, result_shape=float))

        eta, S, upsilon_next = solve_step_subprob(
                    C=state.C,
                    A_data=state.A_data,
                    A_indices=state.A_indices,
                    U=state.U,
                    b=state.b,
                    b_ineq_mask=state.b_ineq_mask,
                    upsilon=state.upsilon,
                    trace_ub=trace_ub,
                    rho=rho,
                    bar_primal_obj=state.bar_primal_obj,
                    z_bar=state.z_bar,
                    tr_X_bar=state.tr_X_bar,
                    y=state.y,
                    V=state.V,
                    n=n,
                    m=m,
                    k=k,
                    subprob_eps=subprob_eps,
                    subprob_max_iters=subprob_max_iters)

        S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
        S_eigvals = jnp.clip(S_eigvals, a_min=0)    # numerical instability handling
        VSV_T_factor = (state.V @ S_eigvecs) * jnp.sqrt(S_eigvals).reshape(1, -1)
        A_operator_VSV_T = apply_A_operator_batched(m, state.A_data, state.A_indices, VSV_T_factor)
        if state.Omega is None:
            X_next = eta * state.X_bar + state.V @ S @ state.V.T
            P_next = None
            primal_obj_next = jnp.trace(state.C @ X_next)
        else:
            X_next = None
            P_next = eta * state.P_bar + VSV_T_factor @ (VSV_T_factor.T @ state.Omega)
            primal_obj_next = eta * state.bar_primal_obj
            primal_obj_next += jnp.trace(VSV_T_factor.T @ (state.C @ VSV_T_factor))
        tr_X_next = eta * state.tr_X_bar + jnp.trace(S)
        z_next = eta * state.z_bar + A_operator_VSV_T
        y_cand = state.y + (1.0 / rho) * (state.b - z_next - upsilon_next)
        y_cand = y_cand - jnp.clip(state.b_ineq_mask * y_cand, a_min=0.0)

        cand_eigvals, cand_eigvecs = eigsh_smallest(
            n=n,
            C=state.C,
            A_data=state.A_data,
            A_indices=state.A_indices,
            adjoint_left_vec=-y_cand,
            q0=state.q0,
            num_desired=k_curr,
            inner_iterations=lanczos_inner_iterations,
            max_restarts=lanczos_max_restarts,
            tolerance=subprob_eps)
        cand_eigvals = -cand_eigvals
        cand_pen_dual_obj = jnp.dot(-state.b, y_cand) + trace_ub*jnp.clip(cand_eigvals[0], a_min=0)
        neg_obj_lb = jnp.clip(
            jnp.dot(-state.b, y_cand)
            + trace_ub*jnp.clip(cand_eigvals[0], a_min=0),
            a_max=state.neg_obj_lb)

        lb_spec_est = compute_lb_spec_est_ipm(
            C=state.C,
            A_data=state.A_data,
            A_indices=state.A_indices,
            U=state.U,
            b=state.b,
            trace_ub=trace_ub,
            bar_primal_obj=state.bar_primal_obj,
            z_bar=state.z_bar,
            tr_X_bar=state.tr_X_bar,
            y=y_cand,
            V=state.V,
            n=n,
            k=k,
            ipm_eps=subprob_eps,
            ipm_max_iters=subprob_max_iters)

        y_next, pen_dual_obj_next = lax.cond(
            beta * (state.pen_dual_obj - lb_spec_est) <= state.pen_dual_obj - cand_pen_dual_obj,
            lambda _: (y_cand, cand_pen_dual_obj),
            lambda _: (state.y, state.pen_dual_obj),
            None)

        y_changed = jnp.logical_or(pen_dual_obj_next == cand_pen_dual_obj, state.y_changed)

        curr_VSV_T_factor = (state.V @ S_eigvecs[:, :k_curr]) * jnp.sqrt(S_eigvals[:k_curr]).reshape(1, -1)
        if state.Omega is None:
            X_bar_next = eta * state.X_bar + curr_VSV_T_factor @ curr_VSV_T_factor.T
            P_bar_next = None
            bar_primal_obj_next = jnp.trace(state.C @ X_bar_next)
        else:
            X_bar_next = None
            P_bar_next = eta * state.P_bar + curr_VSV_T_factor @ (curr_VSV_T_factor.T @ state.Omega)
            bar_primal_obj_next = eta * state.bar_primal_obj
            bar_primal_obj_next += jnp.trace(curr_VSV_T_factor.T @ (state.C @ curr_VSV_T_factor))
        tr_X_bar_next = (eta * state.tr_X_bar + jnp.sum(S_eigvals[:k_curr])).squeeze()
        z_bar_next = eta * state.z_bar
        z_bar_next += apply_A_operator_batched(m, state.A_data, state.A_indices, curr_VSV_T_factor)
        V_next = jnp.concatenate([state.V @ S_eigvecs[:,k_curr:], cand_eigvecs], axis=1)
        V_next, _ = jnp.linalg.qr(
            jnp.concatenate([state.V @ S_eigvecs[:,k_curr:], cand_eigvecs], axis=1))
        
        obj_gap = (primal_obj_next + neg_obj_lb) / (SCALE_C * SCALE_X)
        obj_gap /= 1.0 + jnp.clip(jnp.abs(primal_obj_next), a_min=jnp.abs(neg_obj_lb)) / (SCALE_C * SCALE_X)
        infeas_gap = jnp.linalg.norm((z_next - state.b + upsilon_next) / SCALE_A) / SCALE_X
        infeas_gap /= 1.0 + jnp.linalg.norm((state.b / SCALE_A) / SCALE_X)
        max_infeas = jnp.max(jnp.abs(z_next - state.b + upsilon_next) / SCALE_A) / SCALE_X

        if state.Omega is not None and callback_fn is not None:
            callback_val = callback_fn(
                state.P,
                state.Omega,
                callback_static_args,
                state.callback_nonstatic_args)
        else:
            callback_val = None

        end_time = hcb.call(lambda _: time.time(), arg=0, result_shape=float)
        jax.debug.print("t: {t} - end_time: {end_time} - pen_dual_obj: {pen_dual_obj}"
                        " - cand_pen_dual_obj: {cand_pen_dual_obj} - lb_spec_est: {lb_spec_est}"
                        " - pen_dual_obj_next: {pen_dual_obj_next} - primal_obj: {primal_obj}"
                        " - neg_obj_lb: {neg_obj_lb} - obj_gap: {obj_gap} - infeas_gap: {infeas_gap}"
                        " - max_infeas: {max_infeas} - callback_val: {callback_val}",
                        t=state.t,
                        end_time=end_time,
                        pen_dual_obj=state.pen_dual_obj,
                        cand_pen_dual_obj=cand_pen_dual_obj,
                        lb_spec_est=lb_spec_est,
                        pen_dual_obj_next=pen_dual_obj_next,
                        primal_obj=primal_obj_next,
                        neg_obj_lb=neg_obj_lb,
                        obj_gap=obj_gap,
                        infeas_gap=infeas_gap,
                        max_infeas=max_infeas,
                        callback_val=callback_val)

        return StateStruct(
            t=state.t+1,
            C=state.C,
            A_data=state.A_data,
            A_indices=state.A_indices,
            b=state.b,
            b_ineq_mask=state.b_ineq_mask,
            upsilon=upsilon_next,
            Omega=state.Omega,
            U=state.U,
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
            q0=state.q0,
            callback_nonstatic_args=state.callback_nonstatic_args,
            start_time=state.start_time,
            curr_time=end_time,
            obj_gap=obj_gap,
            infeas_gap=infeas_gap,
            max_infeas=max_infeas,
            primal_obj=primal_obj_next,
            bar_primal_obj=bar_primal_obj_next,
            pen_dual_obj=pen_dual_obj_next,
            lb_spec_est=lb_spec_est,
            neg_obj_lb=neg_obj_lb,
            y_changed=y_changed)


    q0 = jax.random.normal(jax.random.PRNGKey(0), shape=(n,))
    q0 /= jnp.linalg.norm(q0)
    init_eigvals, init_eigvecs = eigsh_smallest(
        n=n,
        C=sdp_state.C,
        A_data=sdp_state.A_data,
        A_indices=sdp_state.A_indices,
        adjoint_left_vec=-sdp_state.y,
        q0=q0,
        num_desired=k,
        inner_iterations=lanczos_inner_iterations,
        max_restarts=lanczos_max_restarts,
        tolerance=subprob_eps)
    init_eigvals = -init_eigvals
    init_pen_dual_obj = jnp.dot(-sdp_state.b, sdp_state.y)
    init_pen_dual_obj += trace_ub*jnp.clip(init_eigvals[0], a_min=0)

    global_start_time = time.time()
    init_state = StateStruct(
        t=jnp.array(0),
        C=sdp_state.C,
        A_data=sdp_state.A_data,
        A_indices=sdp_state.A_indices,
        b=sdp_state.b,
        b_ineq_mask=sdp_state.b_ineq_mask,
        upsilon=jnp.zeros((m,)),
        Omega=sdp_state.Omega,
        U=U,
        X=sdp_state.X,
        tr_X=sdp_state.tr_X,
        X_bar=sdp_state.X,
        tr_X_bar=sdp_state.tr_X,
        P=sdp_state.P,
        P_bar=sdp_state.P,
        z=sdp_state.z,
        z_bar=sdp_state.z,
        y=sdp_state.y,
        V=init_eigvecs,
        q0=q0,
        callback_nonstatic_args=callback_nonstatic_args,
        start_time=global_start_time,
        curr_time=global_start_time,
        obj_gap=1.1*obj_gap_eps,
        infeas_gap=1.1*infeas_gap_eps,
        max_infeas=1.1*max_infeas_eps,
        primal_obj=sdp_state.primal_obj,
        bar_primal_obj=sdp_state.primal_obj,
        pen_dual_obj=init_pen_dual_obj,
        lb_spec_est=jnp.array(0.0),
        neg_obj_lb=jnp.inf,
        y_changed=jnp.array(True))

    final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=max_iters)

    return SDPState(
        C=sdp_state.C,
        A_indices=sdp_state.A_indices,
        A_data=sdp_state.A_data,
        b=sdp_state.b,
        b_ineq_mask=sdp_state.b_ineq_mask,
        X=final_state.X,
        P=final_state.P,
        Omega=sdp_state.Omega,
        y=final_state.y,
        z=final_state.z,
        tr_X=final_state.tr_X,
        primal_obj=final_state.primal_obj,
        SCALE_C=sdp_state.SCALE_C,
        SCALE_X=sdp_state.SCALE_X,
        SCALE_A=sdp_state.SCALE_A)