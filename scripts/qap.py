import jax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from typing import Any, Callable, Tuple

from IPython import embed


def load_and_process_qap(fname: str) -> Tuple[Array, Array]:
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

    C = build_objective_matrix(D, W)
    return C


def build_objective_matrix(D: Array, W: Array) -> BCOO:
    n = D.shape[0]
    sparse_D = BCOO.fromdense(D)
    sparse_W = BCOO.fromdense(W)

    D_indices = sparse_D.indices.reshape(1, sparse_D.nse, 2)
    D_data = sparse_D.data.reshape(1, sparse_D.nse)
    W_indices = sparse_W.indices.reshape(sparse_W.nse, 1, 2)
    W_data = sparse_W.data.reshape(sparse_W.nse, 1)

    W_indices *= n
    W_kron_D_indices = (D_indices + W_indices).reshape(sparse_D.nse * sparse_W.nse, 2)
    W_kron_D_data = (D_data * W_data).reshape(sparse_D.nse * sparse_W.nse,)

    C = BCOO((W_kron_D_data, W_kron_D_indices + 1), shape=(n**2 + 1, n**2 + 1))
    return C


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    DATAFILE = "data/qap/qapdata/chr12a.dat"

    C = load_and_process_qap(DATAFILE)

    embed()
    exit()