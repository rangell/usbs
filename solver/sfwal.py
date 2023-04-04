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