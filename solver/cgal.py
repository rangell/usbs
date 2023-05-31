from collections import namedtuple
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
from functools import partial
import jax
import jax.experimental.host_callback as hcb
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
import time
from typing import Any, Callable, Tuple, Union

from solver.lanczos import eigsh_smallest
from solver.utils import apply_A_operator_slim, apply_A_adjoint_slim

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
    C: BCOO,
    A_data: Array,
    A_indices: Array,
    b: Array,
    Omega: Union[Array, None],
    beta0: float,
    SCALE_C: float,
    SCALE_X: float,
    eps: float,
    max_iters: int,
    line_search: bool,
    lanczos_inner_iterations: int,
    lanczos_max_restarts: int,
    subprob_tol: float,
    callback_fn: Union[Callable[[Array, Array], Array], None]
) -> Tuple[Array, Array]:

    StateStruct = namedtuple(
        "StateStruct",
        ["t",
         "C",
         "A_data",
         "A_indices",
         "b",
         "Omega",
         "X",
         "P",
         "y",
         "z",
         "primal_obj",
         "obj_gap",
         "infeas_gap"])

    @jax.jit
    def cond_func(state: StateStruct) -> bool:
        return jnp.logical_or(state.obj_gap > eps, state.infeas_gap > eps)

    @jax.jit
    def body_func(state: StateStruct) -> StateStruct:
        jax.debug.print("start_time: {time}",
                        time=hcb.call(lambda _: time.time(), arg=0, result_shape=float))
        beta = beta0 * jnp.sqrt(state.t + 1)

        adjoint_left_vec = state.y + beta*(state.z - state.b)

        q0 = jax.random.normal(jax.random.PRNGKey(0), shape=(n,))
        q0 /= jnp.linalg.norm(q0)
        eigvals, eigvecs = eigsh_smallest(
            n=n,
            C=state.C,
            A_data=state.A_data,
            A_indices=state.A_indices,
            adjoint_left_vec=adjoint_left_vec,
            q0=q0,
            num_desired=1,
            inner_iterations=lanczos_inner_iterations,
            max_restarts=lanczos_max_restarts,
            tolerance=subprob_tol)
        # fix trace_ub here to allow for <= trace
        eigvecs = lax.cond(
            (eigvals >= 0).squeeze(), lambda _: 0.0 * eigvecs, lambda _: eigvecs, None)

        min_eigvec = eigvecs[:, 0:1]  # gives the right shape for next line
        X_update_dir = trace_ub * min_eigvec @ min_eigvec.T
        min_eigvec = min_eigvec.reshape(-1,)

        surrogate_dual_gap = state.primal_obj - trace_ub*jnp.dot(min_eigvec, state.C @ min_eigvec)
        surrogate_dual_gap += jnp.dot(adjoint_left_vec, state.z)
        surrogate_dual_gap -= trace_ub * jnp.dot(
            min_eigvec, apply_A_adjoint_slim(
                n, state.A_data, state.A_indices, adjoint_left_vec, min_eigvec))
        obj_gap = surrogate_dual_gap - jnp.dot(state.y, state.z - state.b)
        obj_gap -= 0.5*beta*jnp.linalg.norm(state.z - b)**2
        obj_gap = obj_gap / (SCALE_C * SCALE_X)
        obj_gap /= 1.0 + (jnp.abs(state.primal_obj) / (SCALE_C * SCALE_X))
        infeas_gap = jnp.linalg.norm((state.z - b) / SCALE_X) 
        infeas_gap /= 1.0 + jnp.linalg.norm(b / SCALE_X)
        max_infeas = jnp.max(jnp.abs(state.z - b)) / SCALE_X

        if line_search:
            AH = trace_ub * apply_A_operator_slim(m, state.A_data, state.A_indices, min_eigvec)
            eta = state.primal_obj - trace_ub*jnp.dot(min_eigvec, state.C @ min_eigvec)
            eta += jnp.dot(adjoint_left_vec, state.z - AH)
            eta /= beta * jnp.sum(jnp.square(AH - state.z))
            eta = jnp.clip(eta, a_min=0.0, a_max=1.0)
            #AH = trace_ub * apply_A_operator_slim(m, state.A_data, state.A_indices, min_eigvec)
            #eta = state.primal_obj - trace_ub*jnp.dot(min_eigvec, state.C @ min_eigvec)
            #eta /= (SCALE_C * SCALE_X)
            #eta += jnp.dot(adjoint_left_vec, state.z - AH) / SCALE_X**2
            #eta /= beta * jnp.sum(jnp.square(AH - state.z)) / SCALE_X**2
            #eta = jnp.clip(eta, a_min=0.0, a_max=1.0)
        else:
            eta = 2.0 / (state.t + 2.0)

        if state.Omega is None:
            X_next = (1 - eta) * state.X + eta * trace_ub * X_update_dir
            P_next = None
        else:
            X_next = None
            min_eigvec = min_eigvec.reshape(-1, 1)
            P_next = (1 - eta) * state.P + eta * trace_ub * min_eigvec @ (min_eigvec.T @ state.Omega)
            min_eigvec = min_eigvec.reshape(-1,)

        z_next = (1 - eta) * state.z + eta * trace_ub * apply_A_operator_slim(
            m, state.A_data, state.A_indices, min_eigvec)

        dual_step_size = jnp.clip(
            4 * beta * eta**2 * trace_ub**2 / jnp.sum(jnp.square(z_next - state.b)),
            a_min=0.0,
            a_max=beta0)
        y_next = state.y + dual_step_size * (z_next - state.b)

        primal_obj_next = (1 - eta) * state.primal_obj
        primal_obj_next += eta * trace_ub * jnp.dot(min_eigvec, state.C @ min_eigvec)

        if state.Omega is not None and callback_fn is not None:
            callback_val = callback_fn(state.C / SCALE_C, state.Omega, state.P)
        else:
            callback_val = None

        end_time = hcb.call(lambda _: time.time(), arg=0, result_shape=float)
        jax.debug.print("t: {t} - end_time: {end_time} - primal_obj: {primal_obj} - obj_gap: {obj_gap}"
                        " - infeas_gap: {infeas_gap} - max_infeas: {max_infeas}"
                        " - callback_val: {callback_val} - eta: {eta}",
                        t=state.t,
                        end_time=end_time,
                        primal_obj=state.primal_obj / (SCALE_C * SCALE_X),
                        obj_gap=obj_gap,
                        infeas_gap=infeas_gap,
                        max_infeas=max_infeas,
                        callback_val=callback_val,
                        eta=eta)

        return StateStruct(
            t=state.t+1,
            C=state.C,
            A_data=state.A_data,
            A_indices=state.A_indices,
            b=state.b,
            Omega=state.Omega,
            X=X_next,
            P=P_next,
            y=y_next,
            z=z_next,
            primal_obj=primal_obj_next,
            obj_gap=obj_gap,
            infeas_gap=infeas_gap)

    init_state = StateStruct(
        t=0,
        C=C,
        A_data=A_data,
        A_indices=A_indices,
        b=b,
        Omega=Omega,
        X=X,
        P=P,
        y=y,
        z=z,
        primal_obj=primal_obj,
        obj_gap=1.1*eps,
        infeas_gap=1.1*eps)

    final_state = bounded_while_loop(cond_func, body_func, init_state, max_steps=max_iters)

    #state = init_state
    #for _ in range(1000):
    #    state = body_func(state)

    #embed()
    #exit()

    return (final_state.X,
            final_state.P,
            final_state.y,
            final_state.z,
            final_state.primal_obj,
            trace_ub)