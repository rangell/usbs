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


@partial(jax.jit, static_argnames=["C_matmat", "A_adjoint_batched", "Q_base", "k", "ipm_eps", "ipm_max_iters"])
def solve_quadratic_subproblem(
    C_matmat: Callable[[Array], Array],
    A_adjoint_batched: Callable[[Array, Array], Array],
    Q_base: Callable[[Array], Array],
    U: BCOO,
    b: Array,
    trace_ub: float,
    rho: float,
    bar_primal_obj: Array,
    z_bar: Array,
    tr_X_bar: Array,
    y: Array,
    V: Array,
    k: int,
    ipm_eps: float = 1e-7,
    ipm_max_iters: int = 50
) -> Tuple[Array, Array]:

    svec = lambda mx: U @ mx.reshape(-1)
    svec_inv = lambda vec: (U.T @ vec).reshape(k, k)

    # create problem constants
    tr_X_bar = lax.cond(tr_X_bar > 0.0, lambda _: tr_X_bar, lambda _: trace_ub, None)
    Q_11 = (trace_ub**2 / rho) * Q_base(V)
    q_12 = (trace_ub**2 / (rho * tr_X_bar)) * svec(V.T @ A_adjoint_batched(z_bar, V))
    q_22 = (trace_ub**2 / (rho * tr_X_bar**2)) * jnp.dot(z_bar, z_bar)
    h_1 = trace_ub * svec(V.T @ (C_matmat(V)
                                 - A_adjoint_batched(y, V)
                                 - (A_adjoint_batched(b, V) / rho)))
    h_2 = (trace_ub / tr_X_bar) * (bar_primal_obj - jnp.dot(z_bar, y) - (jnp.dot(z_bar, b) / rho))
    svec_I = svec(jnp.eye(k))

    # initialize all Lagrangian variables
    S_init = 0.99 * (jnp.eye(k) / (k + 1.0))
    eta_init = lax.cond(
        bar_primal_obj == 0.0,
        lambda _: jnp.asarray([0.00001]),
        lambda _: jnp.asarray([0.99 * (1.0 / (k + 1.0))]),
        None)
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
        S_inv = jnp.linalg.inv(ipm_state.S)
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
            max_steps=100,
            base=128)

        step_size = bounded_while_loop(
            lambda step_size: jnp.linalg.eigh(ipm_state.T + step_size * delta_T)[0][0] < 0.0,
            lambda step_size: step_size * 0.8, 
            step_size,
            max_steps=100,
            base=128)

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

        #jax.debug.print("i: {i} - step_size: {step_size} - mu: {mu} - mu_next: {mu_next}"
        #                " - eta: {eta} - eta_next: {eta_next}",
        #                i=ipm_state.i,
        #                mu=ipm_state.mu.squeeze(),
        #                eta=ipm_state.eta.squeeze(),
        #                step_size=step_size,
        #                mu_next=mu_next.squeeze(),
        #                eta_next=eta_next.squeeze())
        
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
        max_steps=ipm_max_iters,
        base=128)

    return ((trace_ub / tr_X_bar) * final_ipm_state.eta.squeeze(),
            trace_ub * final_ipm_state.S)


@partial(jax.jit, static_argnames=["C_matmat", "A_adjoint_batched", "k", "ipm_eps", "ipm_max_iters"])
def compute_lb_spec_est(
    C_matmat: Callable[[Array], Array],
    A_adjoint_batched: Callable[[Array, Array], Array],
    U: BCOO,
    b: Array,
    trace_ub: float,
    bar_primal_obj: Array,
    z_bar: Array,
    tr_X_bar: Array,
    y: Array,
    V: Array,
    k: int,
    ipm_eps: float = 1e-7,
    ipm_max_iters: int = 50
) -> Tuple[Array, Array]:

    svec = lambda mx: U @ mx.reshape(-1)
    svec_inv = lambda vec: (U.T @ vec).reshape(k, k)

    # create problem constants
    tr_X_bar = lax.cond(tr_X_bar > 0.0, lambda _: tr_X_bar, lambda _: trace_ub, None)
    g_1 = trace_ub * svec(V.T @ (C_matmat(V) - A_adjoint_batched(y, V)))
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
            max_steps=100,
            base=128)

        step_size = bounded_while_loop(
            lambda step_size: jnp.linalg.eigh(ipm_state.T + step_size * delta_T)[0][0] < 0.0,
            lambda step_size: step_size * 0.8, 
            step_size,
            max_steps=100,
            base=128)

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

        #jax.debug.print("\ti: {i} - step_size: {step_size} - mu: {mu} - mu_next: {mu_next}"
        #                " - eta: {eta} - eta_next: {eta_next} - delta_eta: {delta_eta}"
        #                " - lb_spec_est: {lb_spec_est} - lb_spec_est_next: {lb_spec_est_next}"
        #                " - obj_gap: {obj_gap}",
        #                i=ipm_state.i,
        #                mu=ipm_state.mu.squeeze(),
        #                eta=ipm_state.eta.squeeze(),
        #                step_size=step_size,
        #                mu_next=mu_next.squeeze(),
        #                eta_next=eta_next.squeeze(),
        #                delta_eta=delta_eta.squeeze(),
        #                lb_spec_est=lb_spec_est,
        #                lb_spec_est_next=lb_spec_est_next,
        #                obj_gap=jnp.abs((lb_spec_est - lb_spec_est_next) / lb_spec_est_next))
    
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
        lambda ipm_state: jnp.logical_and(ipm_state.mu.squeeze() > ipm_eps,
                                          ipm_state.obj_gap > ipm_eps),
        body_func, 
        init_ipm_state,
        max_steps=ipm_max_iters,
        base=128)

    lb_spec_est = jnp.dot(b, y) + jnp.dot(g_1, svec(final_ipm_state.S))
    lb_spec_est += final_ipm_state.eta.squeeze() * g_2
    return -lb_spec_est.squeeze()


def specbm(
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

    @jax.jit
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

    final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=max_iters, base=128)

    return (final_state.X,
            final_state.P,
            final_state.y,
            final_state.z,
            final_state.primal_obj,
            final_state.tr_X)