import jax
import jax.numpy as jnp
from jax._src.typing import Array
from typing import Any, Callable, Tuple

from IPython import embed


def load_qap(fname: str) -> Tuple[Array, Array]:
    """ Return (A, B) problem data from QAP data format. """
    with open(fname, "r") as f:
        datastr = f.read()
        str_n, str_A, str_B, _ = tuple(datastr.split("\n\n"))
        n = int(str_n)
        A = jnp.array([float(v) for v in str_A.split()]).reshape(n, n)
        B = jnp.array([float(v) for v in str_B.split()]).reshape(n, n)
        # by convention, B should be sparser than A
        if jnp.count_nonzero(A) < jnp.count_nonzero(B):
            A, B = B, A

    return A, B


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    DATAFILE = "data/qap/qapdata/chr12a.dat"

    A, B = load_qap(DATAFILE)

    embed()
    exit()