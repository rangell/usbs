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
    C_matvec: Callable[[Array], Array],
    A_operator_slim: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    proj_K: Callable[[Array], Array],
    beta: float,
    k: int,
    SCALE_C: float,
    SCALE_X: float,
    eps: float,
    max_iters: int,
    lanczos_num_iters: int
) -> Tuple[Array, Array]:

    C_matmat = jax.vmap(C_matvec, 1, 1)
    A_adjoint_batched = jax.vmap(A_adjoint_slim, (None, 1), 1)
    A_operator_batched = jax.vmap(A_operator_slim, 1, 1)

    StateStruct = namedtuple(
        "StateStruct",
        ["t", "X", "y", "z", "obj_val", "obj_gap", "infeas_gap"])

    @jax.jit
    def cond_func(state: StateStruct) -> bool:
        return jnp.logical_or(state.obj_gap > eps, state.infeas_gap > eps)

    #@jax.jit
    def body_func(state: StateStruct) -> StateStruct:
        b = proj_K(state.z + (state.y / beta))
        adjoint_left_vec = state.y + beta*(state.z - b)

        _, V = approx_grad_k_min_eigen(
            C_matvec=C_matvec,
            A_adjoint_slim=A_adjoint_slim,
            adjoint_left_vec=adjoint_left_vec,
            n=n,
            k=k,
            num_iters=lanczos_num_iters,
            rng=jax.random.PRNGKey(state.t))

        min_eigvec = V[:, 0:1]  # gives the right shape for next line
        X_update_dir = trace_ub * min_eigvec @ min_eigvec.T
        min_eigvec = min_eigvec.reshape(-1,)

        surrogate_dual_gap = state.obj_val - trace_ub*jnp.dot(min_eigvec, C_matvec(min_eigvec))
        surrogate_dual_gap += jnp.dot(adjoint_left_vec, state.z)
        surrogate_dual_gap -= trace_ub * jnp.dot(min_eigvec, A_adjoint_slim(adjoint_left_vec, min_eigvec))
        obj_gap = surrogate_dual_gap - jnp.dot(state.y, state.z - b)
        obj_gap -= 0.5*beta*jnp.linalg.norm(state.z - b)**2
        obj_gap = obj_gap / (SCALE_C * SCALE_X)
        obj_gap /= 1.0 + (jnp.abs(state.obj_val) / (SCALE_C * SCALE_X))
        infeas_gap = jnp.linalg.norm(state.z - proj_K(state.z)) / SCALE_X
        infeas_gap /= 1.0 + (jnp.linalg.norm(proj_K(state.z)) / SCALE_X)
        max_infeas = jnp.max(jnp.abs(state.z - proj_K(state.z))) / SCALE_X
        jax.debug.print("t: {t} - obj_val: {obj_val} - obj_gap: {obj_gap} -"
                        " infeas_gap: {infeas_gap} - max_infeas: {max_infeas}",
                        t=state.t,
                        obj_val=state.obj_val / (SCALE_C * SCALE_X),
                        obj_gap=obj_gap,
                        infeas_gap=infeas_gap,
                        max_infeas=max_infeas)


        # spectral line search
        # State: (eta, S_eigvals, S_eigvecs, max_value_change)

        step_size = 0.1
        apgd_max_iters = 10

        def apgd(curr: Tuple[float, Array, Array, float]) -> Tuple[float, Array, Array, float]:
            eta, S_eigvals, S_eigvecs, _ = curr
            VSV_T_factor = (V @ S_eigvecs) * jnp.sqrt(S_eigvals).reshape(1, -1)
            A_operator_VSV_T = jnp.sum(A_operator_batched(VSV_T_factor), axis=1)

            grad_S = (V.T @ C_matmat(V)
                      + V.T @ A_adjoint_batched(state.y, V)
                      + V.T @ A_adjoint_batched((eta*state.z) + A_operator_VSV_T - b, V))
            grad_eta = (state.obj_val
                        + jnp.dot(state.y, state.z)
                        + eta * jnp.linalg.norm(state.z)**2
                        + jnp.dot(state.z, A_operator_VSV_T - b))
            S = (S_eigvecs * S_eigvals.reshape(1, -1)) @ S_eigvecs.T

            # TODO: accelerate by adding momentum here
            S_unproj = S - (step_size * grad_S)
            eta_unproj = eta - (step_size * grad_eta)

            S_unproj_eigvals, S_eigvecs = jnp.linalg.eigh(S_unproj)

            eta_index = jnp.sum(S_unproj_eigvals < eta_unproj)
            trace_vals = jnp.insert(S_unproj_eigvals, eta_index, eta_unproj)

            embed()
            exit()

            # project `trace_vals` onto the (k+1)-dim simplex
            # TODO: change this to be the convex hull of the (k+1)-dim simplex
            descend_vals = jnp.flip(trace_vals)
            weighted_vals = (descend_vals
                            + (1.0 / jnp.arange(1, len(descend_vals)+1))
                                * (1 - jnp.cumsum(descend_vals)))
            idx = jnp.sum(weighted_vals > 0) - 1
            offset = weighted_vals[idx] - descend_vals[idx]
            proj_descend_vals = descend_vals + offset
            proj_descend_vals = proj_descend_vals * (proj_descend_vals > 0)
            proj_trace_vals = jnp.flip(proj_descend_vals)

            eta_new = trace_ub * proj_trace_vals[eta_index]
            proj_S_eigvals = trace_ub * jnp.delete(proj_trace_vals, eta_index)
            S_new = (S_eigvecs * proj_S_eigvals.reshape(1, -1)) @ S_eigvecs.T
            max_value_change = jnp.max(
                jnp.append(jnp.abs(S_new - S).reshape(-1,), jnp.abs(eta - eta_new)))

            return (eta_new, proj_S_eigvals, S_eigvecs, max_value_change)

        next = apgd((0, jnp.ones(k) / k * trace_ub, jnp.eye(k), 1.1*eps))

        final_state = bounded_while_loop(
            lambda curr: curr[-1] > eps,
            apgd, 
            (0, jnp.ones(k) / k * trace_ub, jnp.eye(k), 1.1*eps),
            max_steps=apgd_max_iters)

        embed()
        exit()


        eta = 2.0 / (state.t + 2.0)
        X_next = (1-eta)*state.X + eta*X_update_dir
        z_next = (1-eta)*state.z + eta*trace_ub*A_operator_slim(min_eigvec)
        y_next = state.y + beta*(z_next - proj_K(z_next + (state.y / beta)))
        obj_val_next = (1-eta)*state.obj_val + eta*trace_ub*jnp.dot(min_eigvec, C_matvec(min_eigvec))

        return StateStruct(
            t=state.t+1,
            X=X_next,
            y=y_next,
            z=z_next,
            obj_val=obj_val_next,
            obj_gap=obj_gap,
            infeas_gap=infeas_gap)

    init_state = StateStruct(
        t=0,
        X=jnp.zeros((n, n)) * SCALE_X,
        y=jnp.zeros((m,)),
        z=jnp.zeros((m,)),
        obj_val=0.0,
        obj_gap=1.1*eps,
        infeas_gap=1.1*eps)

    state1 = body_func(init_state)

    final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=max_iters)

    return final_state.X, final_state.y