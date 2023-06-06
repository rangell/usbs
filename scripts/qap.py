import jax
import jax.numpy as jnp
from jax._src.typing import Array
from typing import Any, Callable, Tuple

from IPython import embed


def load_qap(fname: str) -> Tuple[Array, Array]:
    """ Return (D, W) problem data from QAP data format. """
    with open(fname, "r") as f:
        datastr = f.read()
        str_n, str_D, str_W, _ = tuple(datastr.split("\n\n"))
        n = int(str_n)
        D = jnp.array([float(v) for v in str_D.split()]).reshape(n, n)
        W = jnp.array([float(v) for v in str_W.split()]).reshape(n, n)
        # by convention, W should be sparser than D
        if jnp.count_nonzero(D) < jnp.count_nonzero(W):
            D, W = W, D

    return D, W


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    DATAFILE = "data/qap/qapdata/chr12a.dat"

    D, W = load_qap(DATAFILE)

    embed()
    exit()