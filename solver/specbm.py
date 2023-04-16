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


def specbm(
    n: int,
    m: int,
    trace_ub: float,
    C_innerprod: Callable[[Array], float],
    C_add: Callable[[Array], Array],
    C_matvec: Callable[[Array], Array],
    A_operator: Callable[[Array], Array],
    A_operator_slim: Callable[[Array], Array],
    A_adjoint: Callable[[Array], Array],
    A_adjoint_slim: Callable[[Array, Array], Array],
    b: Array,
    rho: float,
    beta: float,
    r_curr: int,
    r_past: int,
    SCALE_C: float,
    SCALE_X: float,
    eps: float,
    max_iters: int,
    lanczos_num_iters: int,
    apgd_step_size: float,
    apgd_max_iters: int,
    apgd_eps: float
) -> Tuple[Array, Array]:

    # TODO: fix me
    return jnp.zeros(n, n), jnp.zeros((m,))
