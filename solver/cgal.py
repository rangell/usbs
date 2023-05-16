from collections import namedtuple
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
from functools import partial
import jax
import jax.experimental.host_callback as hcb
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
import time
from typing import Any, Callable, Tuple, Union

from solver.eigen import approx_grad_k_min_eigen

from IPython import embed


def cgal(
    X: Union[Array, None],
    P: Union[Array, None],
    y: Array,
    z: Array,
    primal_obj: float,
    tr_X: float,
    n: int,
    m: int,
    trace_ub: float,
    C_matvec: Callable[[Array], Array],
    A_operator_slim: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    Omega: Union[Array, None],
    b: Array,
    beta0: float,
    SCALE_C: float,
    SCALE_X: float,
    eps: float,
    max_iters: int,
    lanczos_num_iters: int,
    callback_fn: Union[Callable[[Array, Array], Array], None]
) -> Tuple[Array, Array]:

    StateStruct = namedtuple(
        "StateStruct",
        ["t", "X", "P", "y", "z", "primal_obj", "obj_gap", "infeas_gap"])

    @jax.jit
    def cond_func(state: StateStruct) -> bool:
        return jnp.logical_or(state.obj_gap > eps, state.infeas_gap > eps)

    #@jax.jit
    def body_func(state: StateStruct) -> StateStruct:
        jax.debug.print("start_time: {time}",
                        time=hcb.call(lambda _: time.time(), arg=0, result_shape=float))
        beta = beta0 * jnp.sqrt(state.t + 1)

        adjoint_left_vec = state.y + beta*(state.z - b)

        eigvals, eigvecs = approx_grad_k_min_eigen(
            C_matvec=C_matvec,
            A_adjoint_slim=A_adjoint_slim,
            adjoint_left_vec=adjoint_left_vec,
            n=n,
            k=1,
            num_iters=lanczos_num_iters,
            rng=jax.random.PRNGKey(state.t))

        # TODO: fix trace_ub cgal issue
        embed()
        exit()

        min_eigvec = eigvecs[:, 0:1]  # gives the right shape for next line
        # TODO: fix trace_ub here to allow for <= trace
        X_update_dir = trace_ub * min_eigvec @ min_eigvec.T
        min_eigvec = min_eigvec.reshape(-1,)

        surrogate_dual_gap = state.primal_obj - trace_ub*jnp.dot(min_eigvec, C_matvec(min_eigvec))
        surrogate_dual_gap += jnp.dot(adjoint_left_vec, state.z)
        surrogate_dual_gap -= trace_ub * jnp.dot(min_eigvec, A_adjoint_slim(adjoint_left_vec, min_eigvec))
        obj_gap = surrogate_dual_gap - jnp.dot(state.y, state.z - b)
        obj_gap -= 0.5*beta*jnp.linalg.norm(state.z - b)**2
        obj_gap = obj_gap / (SCALE_C * SCALE_X)
        obj_gap /= 1.0 + (jnp.abs(state.primal_obj) / (SCALE_C * SCALE_X))
        infeas_gap = jnp.linalg.norm((state.z - b) / SCALE_X) 
        infeas_gap /= 1.0 + jnp.linalg.norm(b / SCALE_X)
        max_infeas = jnp.max(jnp.abs(state.z - b)) / SCALE_X
        eta = 2.0 / (state.t + 2.0)

        if Omega is None:
            X_next = (1 - eta) * state.X + eta * X_update_dir
            P_next = None
        else:
            X_next = None
            min_eigvec = min_eigvec.reshape(-1, 1)
            P_next = (1 - eta) * state.P + eta * trace_ub * min_eigvec @ (min_eigvec.T @ Omega)
            min_eigvec = min_eigvec.reshape(-1,)

        z_next = (1 - eta) * state.z + eta * trace_ub * A_operator_slim(min_eigvec)

        dual_step_size = jnp.clip(
            4 * beta * eta**2 * trace_ub**2 / jnp.sum(jnp.square(z_next - b)),
            a_min=0.0,
            a_max=beta0)
        y_next = state.y + dual_step_size * (z_next - b)

        primal_obj_next = (1 - eta) * state.primal_obj
        primal_obj_next += eta * trace_ub * jnp.dot(min_eigvec, C_matvec(min_eigvec))

        if Omega is not None and callback_fn is not None:
            callback_val = callback_fn(Omega, state.P)
        else:
            callback_val = None

        end_time = hcb.call(lambda _: time.time(), arg=0, result_shape=float)
        jax.debug.print("t: {t} - end_time: {end_time} - primal_obj: {primal_obj} - obj_gap: {obj_gap}"
                        " - infeas_gap: {infeas_gap} - max_infeas: {max_infeas}"
                        " - callback_val: {callback_val}",
                        t=state.t,
                        end_time=end_time,
                        primal_obj=state.primal_obj / (SCALE_C * SCALE_X),
                        obj_gap=obj_gap,
                        infeas_gap=infeas_gap,
                        max_infeas=max_infeas,
                        callback_val=callback_val)


        return StateStruct(
            t=state.t+1,
            X=X_next,
            P=P_next,
            y=y_next,
            z=z_next,
            primal_obj=primal_obj_next,
            obj_gap=obj_gap,
            infeas_gap=infeas_gap)

    init_state = StateStruct(
        t=0,
        X=X,
        P=P,
        y=y,
        z=z,
        primal_obj=primal_obj,
        obj_gap=1.1*eps,
        infeas_gap=1.1*eps)

    #final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=max_iters)

    state = init_state
    for _ in range(6):
        state = body_func(state)

    embed()
    exit()

    return (final_state.X,
            final_state.P,
            final_state.y,
            final_state.z,
            final_state.primal_obj,
            trace_ub)