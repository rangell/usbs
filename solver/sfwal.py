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


def sfwal(
    n: int,
    m: int,
    trace_ub: float,
    trace_exact: bool,
    C_innerprod: Callable[[Array], float],
    C_add: Callable[[Array], Array],
    C_matvec: Callable[[Array], Array],
    A_operator: Callable[[Array], Array],
    A_operator_slim: Callable[[Array], Array],
    A_adjoint: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    proj_K: Callable[[Array], Array],
    beta: float,
    k: int,
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

    StateStruct = namedtuple(
        "StateStruct",
        ["t", "X", "y", "z", "obj_val", "obj_gap", "infeas_gap"])

    @jax.jit
    def cond_func(state: StateStruct) -> Array:
        return jnp.logical_or(state.obj_gap > eps, state.infeas_gap > eps)

    @jax.jit
    def body_func(state: StateStruct) -> StateStruct:
        b = proj_K(state.z + (state.y / beta))
        adjoint_left_vec = state.y + beta*(state.z - b)

        eigvals, V = approx_grad_k_min_eigen(
            C_matvec=C_matvec,
            A_adjoint_slim=A_adjoint_slim,
            adjoint_left_vec=adjoint_left_vec,
            n=n,
            k=k+1,
            num_iters=lanczos_num_iters,
            rng=jax.random.PRNGKey(state.t))

        # TODO: check trim the last eigenvector for the sake of experimentation?
        V = V[:,:-1]
        min_eigvec = V[:, 0:1]  # gives the right shape for next line
        min_eigvec = min_eigvec.reshape(-1,)
        max_eigval_gap = jnp.max(eigvals[1:] - eigvals[:-1])

        surrogate_dual_gap = state.obj_val - trace_ub*jnp.dot(min_eigvec, C_matvec(min_eigvec))
        surrogate_dual_gap += jnp.dot(adjoint_left_vec, state.z)
        surrogate_dual_gap -= trace_ub * jnp.dot(min_eigvec, A_adjoint_slim(adjoint_left_vec, min_eigvec))
        obj_gap = surrogate_dual_gap - jnp.dot(state.y, state.z - b)
        obj_gap -= 0.5*beta*jnp.linalg.norm(state.z - b)**2
        obj_gap /= (SCALE_C * SCALE_X)
        obj_gap /= 1.0 + (jnp.abs(state.obj_val) / (SCALE_C * SCALE_X))
        infeas_gap = jnp.linalg.norm((state.z - proj_K(state.z)) / SCALE_X) 
        infeas_gap /= 1.0 + jnp.linalg.norm(proj_K(state.z) / SCALE_X)
        max_infeas = jnp.max(jnp.abs(state.z - proj_K(state.z))) / SCALE_X
        jax.debug.print("t: {t} - obj_val: {obj_val} - obj_gap: {obj_gap} -"
                        " infeas_gap: {infeas_gap} - max_infeas: {max_infeas}",
                        t=state.t,
                        obj_val=state.obj_val / (SCALE_C * SCALE_X),
                        obj_gap=obj_gap,
                        infeas_gap=infeas_gap,
                        max_infeas=max_infeas)

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
                      + V.T @ A_adjoint_batched(state.y, V)
                      + beta * V.T @ A_adjoint_batched((eta*state.z) + A_operator_VSV_T - b, V))
            grad_eta = (state.obj_val
                        + jnp.dot(state.y, state.z)
                        + beta * eta * jnp.linalg.norm(state.z)**2
                        + beta * jnp.dot(state.z, A_operator_VSV_T - b))

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
            no_projection_needed = jnp.logical_and(
                jnp.logical_not(trace_exact), jnp.logical_and(jnp.sum(trace_vals) < 1,
                                                              jnp.all(trace_vals > -1e-6)))
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
            max_value_change=jnp.array(1.1*eps))

        final_apgd_state = bounded_while_loop(
            lambda apgd_state: apgd_state.max_value_change > apgd_eps,
            apgd, 
            init_apgd_state,
            max_steps=apgd_max_iters)

        # update variables
        S_eigvals, S_eigvecs = jnp.linalg.eigh(final_apgd_state.S_curr)
        S_eigvals = jnp.where(S_eigvals < 0, 0, S_eigvals)
        VSV_T_factor = (V @ S_eigvecs) * jnp.sqrt(S_eigvals).reshape(1, -1)
        A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)
        X_next = final_apgd_state.eta_curr*state.X + V @ final_apgd_state.S_curr @ V.T
        z_next = final_apgd_state.eta_curr*state.z + A_operator_VSV_T
        y_next = state.y + 0.5*beta*(z_next - proj_K(z_next + (state.y / beta)))
        obj_val_next = final_apgd_state.eta_curr*state.obj_val
        obj_val_next += jnp.trace(VSV_T_factor.T @ C_matmat(VSV_T_factor))

        return StateStruct(
            t=state.t+1,
            X=X_next,
            y=y_next,
            z=z_next,
            obj_val=obj_val_next,
            obj_gap=obj_gap,
            infeas_gap=infeas_gap)
    

    #init_state = StateStruct(
    #    t=0,
    #    X=jnp.zeros((n, n)),
    #    y=jnp.zeros((m,)),
    #    z=jnp.zeros((m,)),
    #    obj_val=0.0,
    #    obj_gap=1.1*eps,
    #    infeas_gap=1.1*eps)

    #X_0 = jax.random.normal(jax.random.PRNGKey(0), shape=(n,n))
    #X_0 /= jnp.trace(X_0)
    #y_0 = jax.random.normal(jax.random.PRNGKey(1), shape=(m,))
    X_0 = jnp.ones((n, n)) * SCALE_X * trace_ub
    #X_0 = jnp.zeros((n, n)) * SCALE_X * trace_ub
    z_0 = A_operator(X_0)
    y_0 = jnp.zeros((m,))
    init_state = StateStruct(
        t=0,
        X=X_0,
        y=y_0,
        z=z_0,
        obj_val=C_innerprod(X_0),
        obj_gap=1.1*eps,
        infeas_gap=1.1*eps)

    #state1 = body_func(init_state)

    final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=max_iters)
    embed()
    exit()

    return final_state.X, final_state.y