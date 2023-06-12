from collections import namedtuple
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
import jax.numpy as jnp
from jax._src.typing import Array

from IPython import embed

def munkres(cost_mx: Array) -> Array:
    n = cost_mx.shape[0]
    # create the working matrix
    M = jnp.copy(cost_mx)

    # row and column reduction
    M = M - jnp.min(M, axis=1).reshape(-1, 1)
    M = M - jnp.min(M, axis=0).reshape(1, -1)

    def get_concat_cover_mask(M: Array) -> Array:
        # test for optimal assignment
        zeros_mask = (M == 0)
        row_counts = jnp.sum(zeros_mask, axis=1)
        col_counts = jnp.sum(zeros_mask, axis=0)

        lte_mask = row_counts.reshape(-1, 1) <= col_counts.reshape(1, -1)
        adj_row_counts = row_counts - jnp.sum(lte_mask, axis=1)
        adj_col_counts = col_counts - jnp.sum(~lte_mask, axis=0)

        adj_counts = jnp.concatenate([adj_row_counts, adj_col_counts])
        return (adj_counts > 0)
    
    def cond_func(M: Array) -> bool:
        return jnp.sum(get_concat_cover_mask(M)) != float(n)

    def body_func(M: Array) -> Array:
        concat_cover_mask = get_concat_cover_mask(M)
        row_cover_mask = concat_cover_mask[:n]
        col_cover_mask = concat_cover_mask[n:]

        # shift zeros
        cover_mask = row_cover_mask.reshape(-1, 1) | col_cover_mask.reshape(1, -1)
        intersect_mask = row_cover_mask.reshape(-1, 1) & col_cover_mask.reshape(1, -1)
        min_uncovered_val = jnp.min(jnp.where(~cover_mask, M, jnp.max(M)))
        M = jnp.where(~cover_mask, M - min_uncovered_val, M)
        M = jnp.where(intersect_mask, M + min_uncovered_val, M)
        return M

    M = bounded_while_loop(cond_func, body_func, M, max_steps=n)

    # TODO: extract minimal assignment
    embed()
    exit()



if __name__ == "__main__":
    C = jnp.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    assignment = munkres(C)

    print("C: ", C)
    print("assignment: ", assignment)



