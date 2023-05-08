from collections import namedtuple
import cvxpy as cp
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
from functools import partial
import jax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from jax import lax
from typing import Any, Callable, Tuple

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
    ipm_max_iters: int = 100
) -> Tuple[Array, Array]:

    svec = lambda mx: U @ mx.reshape(-1)
    svec_inv = lambda vec: (U.T @ vec).reshape(k, k)

    # create problem constants
    Q_11 = (trace_ub**2 / rho) * Q_base(V)
    q_12 = (trace_ub**2 / (rho * tr_X_bar)) * svec(V.T @ A_adjoint_batched(z_bar, V))
    q_22 = (trace_ub**2 / (rho * tr_X_bar**2)) * jnp.dot(z_bar, z_bar)
    h_1 = trace_ub * svec(V.T @ (C_matmat(V)
                                 - A_adjoint_batched(y, V)
                                 - (A_adjoint_batched(b, V) / rho)))
    h_2 = (trace_ub / tr_X_bar) * (bar_primal_obj - jnp.dot(z_bar, y) - (jnp.dot(z_bar, b) / rho))
    svec_I = svec(jnp.eye(k))

    # initialize all Lagrangian variables
    S_init = 0.9999 * (jnp.eye(k) / (k + 1.0))
    eta_init = jnp.asarray([0.9999 * (1.0 / (k + 1.0))])
    T_init = svec_inv(Q_11 @ svec(S_init) + eta_init * q_12 + h_1)
    zeta_init = jnp.dot(q_12, svec(S_init)) + eta_init * q_22 + h_2
    omega_init = jnp.asarray([-1.00001 * jnp.min(jnp.append(jnp.diag(T_init), zeta_init))])
    T_init += omega_init * jnp.eye(k)
    zeta_init += omega_init
    mu_init = (jnp.dot(svec(S_init), svec(T_init))
               + eta_init * zeta_init
               + omega_init*(1 - jnp.dot(svec_I, svec(S_init)) - eta_init)) / (k + 2.0)

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

        step_size_numers = jnp.concatenate(
            [jnp.diag(ipm_state.S),
             ipm_state.eta,
             jnp.diag(ipm_state.T),
             ipm_state.zeta,
             ipm_state.omega])
        step_size_denoms = jnp.concatenate(
            [jnp.diag(delta_S),
             delta_eta,
             jnp.diag(delta_T),
             delta_zeta,
             delta_omega])

        step_size = jnp.max(
            jnp.clip(-jnp.clip(step_size_numers / step_size_denoms, a_max=0.0), a_max=1.0))
        step_size = 0.9999 * lax.cond(step_size == 0.0, lambda _: 1.0, lambda _: step_size, None)

        S_next = ipm_state.S + step_size * delta_S
        eta_next = ipm_state.eta + step_size * delta_eta
        T_next = ipm_state.T + step_size * delta_T
        zeta_next = ipm_state.zeta + step_size * delta_zeta
        omega_next = ipm_state.omega + step_size * delta_omega

        mu_next = (jnp.dot(svec(S_next), svec(T_next))
                   + eta_next * zeta_next
                   + omega_next*(1 - jnp.dot(svec_I, svec(S_next)) - eta_next)) / (k + 2.0)
        mu_next *= lax.cond(step_size > 0.2, lambda _: 0.5 - 0.4 * step_size**2, lambda _: 1.0, None)
        mu_next = jnp.clip(mu_next, a_max=ipm_state.mu)

        jax.debug.print("i: {i} - step_size: {step_size} - mu: {mu} - mu_next: {mu_next}"
                        " - eta: {eta} - eta_next: {eta_next}",
                        i=ipm_state.i,
                        mu=ipm_state.mu.squeeze(),
                        eta=ipm_state.eta.squeeze(),
                        step_size=step_size,
                        mu_next=mu_next.squeeze(),
                        eta_next=eta_next.squeeze())
    
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

    return final_ipm_state.eta.squeeze(), trace_ub * final_ipm_state.S


@partial(jax.jit, static_argnames=["C_matmat", "A_operator_batched", "A_adjoint_batched"])
def compute_lb_spec_est(
    C_matmat: Callable[[Array], Array],
    A_operator_batched: Callable[[Array], Array],
    A_adjoint_batched: Callable[[Array, Array], Array],
    b: Array,
    trace_ub: float,
    bar_primal_obj: Array,
    z_bar: Array,
    tr_X_bar: Array,
    y: Array,
    V: Array,
) -> Array:
    trace_ratio_X_bar = lax.cond(tr_X_bar > 0.0, lambda _: trace_ub / tr_X_bar, lambda _: 1.0, None)
    grad_S = trace_ub * V.T @ C_matmat(V) - trace_ub * V.T @ A_adjoint_batched(y, V)
    grad_eta = trace_ratio_X_bar * (bar_primal_obj - jnp.dot(y, z_bar))

    # clip the gradients
    grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_S.flatten())) + grad_eta**2)
    grad_S /= grad_norm
    grad_eta /= grad_norm

    S_unproj = -grad_S
    eta_unproj = -grad_eta

    S_unproj_eigvals, S_eigvecs = jnp.linalg.eigh(S_unproj)
    trace_vals = jnp.append(S_unproj_eigvals, eta_unproj)

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

    proj_trace_vals = proj_simplex(trace_vals)
    eta_next = proj_trace_vals[-1]
    proj_S_eigvals = proj_trace_vals[:-1]
    S_next = (S_eigvecs * trace_ub * proj_S_eigvals.reshape(1, -1)) @ S_eigvecs.T

    VSV_T_factor = (V @ (S_eigvecs)) * jnp.sqrt(trace_ub * proj_S_eigvals).reshape(1, -1)
    A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)
    lb_spec_est = (jnp.dot(-b, y) + eta_next*jnp.dot(y, z_bar) + jnp.dot(y, A_operator_VSV_T)
                   - eta_next*bar_primal_obj - jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor)))
    return eta_next, S_next, lb_spec_est 


@partial(jax.jit, static_argnames=["C_matmat", "A_operator_batched", "A_adjoint_batched", "k", "apgd_max_iters", "apgd_eps"])
def compute_lb_spec_est_slow(
    C_matmat: Callable[[Array], Array],
    A_operator_batched: Callable[[Array], Array],
    A_adjoint_batched: Callable[[Array, Array], Array],
    b: Array,
    trace_ub: float,
    bar_primal_obj: Array,
    z_bar: Array,
    tr_X_bar: Array,
    y: Array,
    V: Array,
    k: int,
    apgd_step_size: float,
    apgd_max_iters: int,
    apgd_eps: float
) -> Tuple[Array, Array, Array, Array]:

    APGDState = namedtuple(
        "APGDState",
        ["i",
         "eta_curr",
         "eta_past",
         "S_curr",
         "S_curr_eigvals",
         "S_curr_eigvecs",
         "S_past",
         "max_value_change"])

    # precompute static parts of the gradients
    trace_ratio_X_bar = lax.cond(tr_X_bar > 0.0, lambda _: trace_ub / tr_X_bar, lambda _: 1.0, None)
    grad_S = trace_ub * V.T @ C_matmat(V) - trace_ub * V.T @ A_adjoint_batched(y, V)
    grad_eta = trace_ratio_X_bar * (bar_primal_obj - jnp.dot(y, z_bar))

    # clip the gradients
    grad_norm = jnp.sqrt(jnp.sum(jnp.square(grad_S.flatten())) + grad_eta**2)
    grad_norm = jnp.max(grad_norm, initial=1.0) 
    grad_S /= grad_norm
    grad_eta /= grad_norm

    @jax.jit
    def apgd(apgd_state: APGDState) -> APGDState:
        momentum = apgd_state.i / (apgd_state.i + 3)  # set this to 0.0 for standard PGD
        #momentum = 0.0
        S = apgd_state.S_curr +  momentum * (apgd_state.S_curr - apgd_state.S_past)
        eta = apgd_state.eta_curr + momentum * (apgd_state.eta_curr - apgd_state.eta_past)
        S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
        S_eigvals = jnp.clip(S_eigvals, a_min=0)    # numerical instability handling

        # compute unprojected steps
        S_unproj = S - (apgd_step_size * grad_S)
        eta_unproj = eta - (apgd_step_size * grad_eta)

        S_unproj_eigvals, S_eigvecs = jnp.linalg.eigh(S_unproj)
        trace_vals = jnp.append(S_unproj_eigvals, eta_unproj)

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
        proj_S_eigvals = proj_trace_vals[:-1]
        S_next = (S_eigvecs * proj_S_eigvals.reshape(1, -1)) @ S_eigvecs.T
        max_value_change = jnp.max(
            jnp.append(jnp.abs(apgd_state.S_curr - S_next).reshape(-1,),
                        jnp.abs(apgd_state.eta_curr - eta_next)))

        return APGDState(
            i=apgd_state.i+1,
            eta_curr=eta_next,
            eta_past=apgd_state.eta_curr,
            S_curr=S_next,
            S_curr_eigvals=proj_S_eigvals,
            S_curr_eigvecs=S_eigvecs,
            S_past=apgd_state.S_curr,
            max_value_change=max_value_change)

    init_apgd_state = APGDState(
        i=0.0,
        eta_curr=jnp.array(0.0),
        eta_past=jnp.array(0.0),
        S_curr=jnp.zeros((k,k)),
        S_curr_eigvals=jnp.zeros((k,)),
        S_curr_eigvecs=jnp.eye(k),
        S_past=jnp.zeros((k,k)),
        max_value_change=jnp.array(1.1*apgd_eps))

    final_apgd_state = bounded_while_loop(
        lambda apgd_state: apgd_state.max_value_change > apgd_eps,
        apgd, 
        init_apgd_state,
        max_steps=apgd_max_iters)

    VSV_T_factor = (V @ (final_apgd_state.S_curr_eigvecs)) * jnp.sqrt(trace_ub * final_apgd_state.S_curr_eigvals).reshape(1, -1)
    A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)
    lb_spec_est = (jnp.dot(-b, y) + final_apgd_state.eta_curr*jnp.dot(y, z_bar) + jnp.dot(y, A_operator_VSV_T)
                   - final_apgd_state.eta_curr*bar_primal_obj - jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor)))

    #jax.debug.print("jnp.dot(-b, y): {val}", val=jnp.dot(-b, y))
    #jax.debug.print("eta*jnp.dot(y, z_bar): {val}", val=final_apgd_state.eta_curr*jnp.dot(y, z_bar))
    #jax.debug.print("jnp.dot(y, A_operator_VSV_T): {val}", val=jnp.dot(y, A_operator_VSV_T))
    #jax.debug.print("-eta*bar_primal_obj: {val}", val=-final_apgd_state.eta_curr*bar_primal_obj)
    #jax.debug.print("-jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor)): {val}", val=-jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor)))

    return final_apgd_state.eta_curr, trace_ub * final_apgd_state.S_curr, y, A_operator_VSV_T, lb_spec_est 


def specbm(
    X: Array,
    y: Array,
    z: Array,
    primal_obj: float,
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
    Q_base: Callable[[Array], Array],
    U: BCOO,
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

    C_matmat = jax.vmap(C_matvec, 1, 1)
    A_adjoint_batched = jax.vmap(A_adjoint_slim, (None, 1), 1)
    A_operator_batched = jax.vmap(A_operator_slim, 1, 1)

    k = k_curr + k_past

    # State:
    #   X
    #   X_bar
    #   tr_X_bar
    #   z
    #   z_bar
    #   y
    #   V
    #   pen_dual_obj, i.e. f(y)
    #   primal_obj, i.e. <C, X>
    #   bar_primal_obj, i.e. <C, X_bar>    
    #   lb_spec_est, i.e. f_hat(y, X_bar)

    StateStruct = namedtuple(
        "StateStruct",
        ["t", 
         "X",
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

    #@jax.jit
    def body_func(state: StateStruct) -> StateStruct:

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

        #embed()
        #exit()

        #del S_
        #del eta_

        #S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
        #S_eigvals = jnp.clip(S_eigvals, a_min=0)    # numerical instability handling

        ################################################################################

        S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
        S_eigvals = jnp.clip(S_eigvals, a_min=0)    # numerical instability handling
        VSV_T_factor = (state.V @ S_eigvecs) * jnp.sqrt(S_eigvals).reshape(1, -1)
        A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)
        X_next = eta * state.X_bar + state.V @ S @ state.V.T
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
            rng=jax.random.PRNGKey(0))
        cand_eigvals = -cand_eigvals
        cand_pen_dual_obj = jnp.dot(-b, y_cand) + trace_ub*jnp.clip(cand_eigvals[0], a_min=0)

        #####################################################################################

        # TODO: write IPM for `lb_spec_est`

        embed()
        exit()

        X = state.X_bar
        y = y_cand
        V = state.V

        S_ = cp.Variable((k,k), symmetric=True)
        eta_ = cp.Variable((1,))
        constraints = [S_ >> 0]
        constraints += [eta_ >= 0]
        constraints += [cp.trace(S_) + eta_*cp.trace(X) <= trace_ub]
        prob = cp.Problem(
            cp.Maximize(-y @ b + cp.trace((eta_ * X + V @ S_ @ V.T) @ (cp.diag(y) - C.todense()))),
            constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        _S_eigvals, _S_eigvecs = jnp.linalg.eigh(S_.value)
        _S_eigvals = jnp.clip(_S_eigvals, a_min=0.0)
        VSV_T_factor = (V @ (_S_eigvecs)) * jnp.sqrt(_S_eigvals).reshape(1, -1)
        A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)
        lb_spec_est = (jnp.dot(-b, y) + eta_.value*jnp.dot(y_cand, state.z_bar) + jnp.dot(y_cand, A_operator_VSV_T)
                    - eta_.value*state.bar_primal_obj - jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor)))
        lb_spec_est = jnp.sum(lb_spec_est)

        #####################################################################################

        #__eta, __S, lb_spec_est_nope = compute_lb_spec_est(
        #    C_matmat=C_matmat,
        #    A_operator_batched=A_operator_batched,
        #    A_adjoint_batched=A_adjoint_batched,
        #    b=b,
        #    trace_ub=trace_ub,
        #    bar_primal_obj=state.bar_primal_obj,
        #    tr_X_bar=state.tr_X_bar,
        #    z_bar=state.z_bar,
        #    y=y_cand,
        #    V=state.V)
        
        ## compute the slower version as a sanity check
        #_eta, _S, _, _, lb_spec_est = compute_lb_spec_est_slow(
        #    C_matmat=C_matmat,
        #    A_operator_batched=A_operator_batched,
        #    A_adjoint_batched=A_adjoint_batched,
        #    b=b,
        #    trace_ub=trace_ub,
        #    bar_primal_obj=state.bar_primal_obj,
        #    tr_X_bar=state.tr_X_bar,
        #    z_bar=state.z_bar,
        #    y=y_cand,
        #    V=state.V,
        #    k=k,
        #    apgd_step_size=apgd_step_size,
        #    apgd_max_iters=apgd_max_iters,
        #    apgd_eps=apgd_eps)

        y_next, pen_dual_obj_next = lax.cond(
            beta * (state.pen_dual_obj - lb_spec_est) <= state.pen_dual_obj - cand_pen_dual_obj,
            lambda _: (y_cand, cand_pen_dual_obj),
            lambda _: (state.y, state.pen_dual_obj),
            None)

        curr_VSV_T_factor = (state.V @ S_eigvecs[:, :k_curr]) * jnp.sqrt(S_eigvals[:k_curr]).reshape(1, -1)
        X_bar_next = eta * state.X_bar + curr_VSV_T_factor @ curr_VSV_T_factor.T
        z_bar_next =  eta * state.z_bar + jnp.sum(A_operator_batched(curr_VSV_T_factor), axis=1)
        V_next = jnp.concatenate([state.V @ S_eigvecs[:,k_curr:], cand_eigvecs], axis=1)
        bar_primal_obj_next = eta * state.bar_primal_obj
        bar_primal_obj_next += jnp.trace(curr_VSV_T_factor.T @ C_matmat(curr_VSV_T_factor))
        
        #infeas_gap = jnp.linalg.norm(z_next - b) 
        #infeas_gap /= 1.0 + jnp.linalg.norm(b)
        #max_infeas = jnp.max(jnp.abs(z_next - b)) 
        jax.debug.print("t: {t} - pen_dual_obj: {pen_dual_obj} - cand_pen_dual_obj: {cand_pen_dual_obj}"
                        " - lb_spec_est: {lb_spec_est} - pen_dual_obj_next: {pen_dual_obj_next}",
                        t=state.t,
                        pen_dual_obj=state.pen_dual_obj,
                        cand_pen_dual_obj=cand_pen_dual_obj,
                        lb_spec_est=lb_spec_est,
                        pen_dual_obj_next=pen_dual_obj_next)

        return StateStruct(
            t=state.t+1,
            X=X_next,
            X_bar=X_bar_next,
            tr_X_bar=jnp.trace(X_bar_next),  # TODO: implement space efficient version
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
        k=1,
        num_iters=lanczos_num_iters,
        rng=jax.random.PRNGKey(0))
    prev_eigvals = -prev_eigvals
    pen_dual_obj = jnp.dot(-b, y) + trace_ub*jnp.clip(prev_eigvals[0], a_min=0)
    # TODO: use `prev_eigvecs` for initializing `V`?

    init_state = StateStruct(
        t=0,
        X=X,
        X_bar=X,
        tr_X_bar=jnp.trace(X),
        z=z,
        z_bar=z,
        y=y,
        V=V,
        primal_obj=primal_obj,
        bar_primal_obj=primal_obj,
        pen_dual_obj=pen_dual_obj,
        lb_spec_est=0.0)

    #final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=10)
    #state = init_state
    #for _ in range(11):
    #    state = body_func(state)

    #import pickle
    #with open("state.pkl", "wb") as f:
    #    pickle.dump(tuple(state), f)

    #embed()
    #exit()

    import pickle
    with open("state.pkl", "rb") as f:
        state = pickle.load(f)
        state = StateStruct(*state)

    next_state = body_func(state)

    embed()
    exit()



    # TODO: check `solve_subproblem` against SCS and MOSEK

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

    embed()
    exit()

    # TODO: fix to return all things needed for warm-start
    return jnp.zeros(n, n), jnp.zeros((m,))
