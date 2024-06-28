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
from utils.common import SDPState

from IPython import embed


def cgal(
    sdp_state: SDPState,
    n: int,
    m: int,
    trace_ub: float,
    beta0: float,
    max_iters: int,
    max_time: float,
    obj_gap_eps: float,
    infeas_gap_eps: float,
    max_infeas_eps: float,
    lanczos_inner_iterations: int,
    lanczos_max_restarts: int,
    subprob_eps: float,
    callback_fn: Union[Callable[[Array, Array], Array], None],
    callback_static_args: bytes,
    callback_nonstatic_args: Any
) -> Tuple[Array, Array]:

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
         "Omega",
         "X",
         "P",
         "y",
         "z",
         "tr_X",
         "callback_nonstatic_args",
         "start_time",
         "curr_time",
         "obj_gap",
         "infeas_gap",
         "max_infeas",
         "primal_obj"])

    @jax.jit
    def cond_func(state: StateStruct) -> bool:
        # NOTE: bounded_while_loop takes care of max_iters
        return jnp.logical_or(
            state.t == 0,
            jnp.logical_and(
                state.curr_time - state.start_time < max_time,
                jnp.logical_or(
                    state.obj_gap > obj_gap_eps,
                    jnp.logical_or(state.infeas_gap > infeas_gap_eps,
                                   state.max_infeas > max_infeas_eps))))

    @jax.jit
    def body_func(state: StateStruct) -> StateStruct:
        jax.debug.print("start_time: {time}",
                        time=jax.pure_callback(lambda : time.time(), result_shape_dtypes=jnp.array(0.0)))
        beta = beta0 * jnp.sqrt(state.t + 1)

        proj_b = (1 - state.b_ineq_mask) * state.b
        proj_b += state.b_ineq_mask * jnp.minimum(state.z + (1.0 / beta) * state.y, state.b)

        adjoint_left_vec = state.y + beta*(state.z - proj_b)

        q0 = jax.random.normal(jax.random.PRNGKey(0), shape=(n,))
        q0 /= jnp.linalg.norm(q0)
        eigvals, eigvecs = eigsh_smallest(
            n=n,
            C=-state.C,
            A_data=state.A_data,
            A_indices=state.A_indices,
            adjoint_left_vec=adjoint_left_vec,
            q0=q0,
            num_desired=1,
            inner_iterations=lanczos_inner_iterations,
            max_restarts=lanczos_max_restarts,
            tolerance=subprob_eps)

        # fix trace_ub here to allow for <= trace
        eigvecs = lax.cond(
            (eigvals >= 0).squeeze(), lambda _: 0.0 * eigvecs, lambda _: eigvecs, None)

        min_eigvec = eigvecs[:, 0:1]  # gives the right shape for next line
        X_update_dir = min_eigvec @ min_eigvec.T
        min_eigvec = min_eigvec.reshape(-1,)

        surrogate_dual_gap = trace_ub*jnp.dot(min_eigvec, state.C @ min_eigvec) - state.primal_obj
        surrogate_dual_gap += jnp.dot(adjoint_left_vec, state.z)
        surrogate_dual_gap -= trace_ub * jnp.dot(
            min_eigvec, apply_A_adjoint_slim(
                n, state.A_data, state.A_indices, adjoint_left_vec, min_eigvec))
        obj_gap = surrogate_dual_gap - jnp.dot(state.y, state.z - proj_b)
        obj_gap -= 0.5*beta*jnp.linalg.norm(state.z - proj_b)**2
        obj_gap = obj_gap / (SCALE_C * SCALE_X)
        obj_gap /= 1.0 + (jnp.abs(state.primal_obj) / (SCALE_C * SCALE_X))

        proj_b = (1 - state.b_ineq_mask) * state.b
        proj_b += state.b_ineq_mask * jnp.minimum(state.z, state.b)
        infeas_gap = jnp.linalg.norm(((state.z - proj_b) / SCALE_A) / SCALE_X) 
        infeas_gap /= 1.0 + jnp.linalg.norm((proj_b / SCALE_A) / SCALE_X)
        max_infeas = jnp.max(jnp.abs((state.z - proj_b) / SCALE_A)) / SCALE_X

        eta = 2.0 / (state.t + 2.0)

        if state.Omega is None:
            X_next = (1 - eta) * state.X + eta * trace_ub * X_update_dir
            P_next = None
        else:
            X_next = None
            min_eigvec = min_eigvec.reshape(-1, 1)
            P_next = (1 - eta) * state.P + eta * trace_ub * min_eigvec @ (min_eigvec.T @ state.Omega)
            min_eigvec = min_eigvec.reshape(-1,)
        
        tr_X_next = (1 - eta) * state.tr_X + eta * trace_ub * jnp.linalg.norm(min_eigvec)**2
        z_next = (1 - eta) * state.z + eta * trace_ub * apply_A_operator_slim(
            m, state.A_data, state.A_indices, min_eigvec)

        # update beta before dual step
        beta = beta0 * jnp.sqrt(state.t + 2)
        proj_b = (1 - state.b_ineq_mask) * state.b
        proj_b += state.b_ineq_mask * jnp.minimum(z_next + (1.0 / beta) * state.y, state.b)
        dual_step_size = jnp.clip(
            4 * beta * eta**2 * trace_ub**2 / jnp.sum(jnp.square(z_next - proj_b)),
            a_min=0.0,
            a_max=beta0)
        y_next = state.y + dual_step_size * (z_next - proj_b)

        primal_obj_next = (1 - eta) * state.primal_obj
        primal_obj_next += eta * trace_ub * jnp.dot(min_eigvec, state.C @ min_eigvec)

        if state.Omega is not None and callback_fn is not None:
            callback_val = callback_fn(
                state.P,
                state.Omega,
                callback_static_args,
                state.callback_nonstatic_args)
        else:
            callback_val = None

        end_time = jax.pure_callback(lambda : time.time(), result_shape_dtypes=jnp.array(0.0))
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
            C=state.C,
            A_data=state.A_data,
            A_indices=state.A_indices,
            b=state.b,
            b_ineq_mask=state.b_ineq_mask,
            Omega=state.Omega,
            X=X_next,
            P=P_next,
            y=y_next,
            z=z_next,
            tr_X=tr_X_next,
            callback_nonstatic_args=state.callback_nonstatic_args,
            start_time=state.start_time,
            curr_time=end_time,
            obj_gap=obj_gap,
            infeas_gap=infeas_gap,
            max_infeas=max_infeas,
            primal_obj=primal_obj_next)

    global_start_time = time.time()
    init_state = StateStruct(
        t=0,
        C=sdp_state.C,
        A_data=sdp_state.A_data,
        A_indices=sdp_state.A_indices,
        b=sdp_state.b,
        b_ineq_mask=sdp_state.b_ineq_mask,
        Omega=sdp_state.Omega,
        X=sdp_state.X,
        P=sdp_state.P,
        y=sdp_state.y,
        z=sdp_state.z,
        tr_X=sdp_state.tr_X,
        callback_nonstatic_args=callback_nonstatic_args,
        start_time=global_start_time,
        curr_time=global_start_time,
        obj_gap=1.1*obj_gap_eps,
        infeas_gap=1.1*infeas_gap_eps,
        max_infeas=1.1*max_infeas_eps,
        primal_obj=sdp_state.primal_obj)

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
