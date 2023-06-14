from collections import namedtuple
from functools import partial
from equinox.internal._loop.bounded import bounded_while_loop # type: ignore
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array

from IPython import embed


@partial(jax.jit, static_argnames=["n"])
def munkres(n: int, cost_mx: Array) -> Array:
    """
    Implementation adapted from: http://csclab.murraystate.edu/bob.pilgrim/445/munkres.html
    """

    # create the working matrix
    M = jnp.copy(cost_mx)

    # Step 1: row reduction
    M = M - jnp.min(M, axis=1).reshape(-1, 1)

    StateStruct = namedtuple("StateStruct", ["M", "mask_mx", "row_cover", "col_cover"])

    # initial starring of zeros
    def initial_star_body(i: int, state: StateStruct) -> StateStruct:
        row, col = i // n, i % n
        star_entry = jnp.logical_and(
            M[row, col] == 0, jnp.logical_and(state.row_cover[row] == 0, state.col_cover[col] == 0))
        next_mask_mx = jnp.copy(state.mask_mx)
        next_mask_mx = next_mask_mx.at[row, col].set(star_entry)
        next_row_cover = jnp.copy(state.row_cover)
        next_row_cover = next_row_cover.at[row].set(
            jnp.clip(star_entry, a_min=state.row_cover.at[row].get()))
        next_col_cover = jnp.copy(state.col_cover)
        next_col_cover = next_col_cover.at[col].set(
            jnp.clip(star_entry, a_min=state.col_cover.at[col].get()))
        return StateStruct(state.M, next_mask_mx, next_row_cover, next_col_cover)
    # Step 2
    init_state = StateStruct(M, jnp.zeros_like(M), jnp.zeros((n,), dtype=bool), jnp.zeros((n,), dtype=bool))
    final_state = lax.fori_loop(0, n**2, initial_star_body, init_state)

    # hard assignment body_func helper functions
    def shift_zeros(state: StateStruct) -> StateStruct:
        # Step 6
        cover_mask = state.row_cover.reshape(-1, 1) | state.col_cover.reshape(1, -1)
        min_uncovered_val = jnp.min(jnp.where(~cover_mask, state.M, jnp.max(state.M)))
        sub_min_mask = jnp.ones_like(state.row_cover).reshape(-1, 1) & (~state.col_cover).reshape(1, -1)
        add_min_mask = state.row_cover.reshape(-1, 1) & jnp.ones_like(state.col_cover).reshape(1, -1)
        M_next = jnp.where(sub_min_mask, state.M - min_uncovered_val, state.M)
        M_next = jnp.where(add_min_mask, M_next + min_uncovered_val, M_next)
        return StateStruct(M_next, state.mask_mx, state.row_cover, state.col_cover)

    def adjust_cover(state: StateStruct, row: int, col: int) -> StateStruct:
        # part of Step 4
        col = jnp.argmax(state.mask_mx[row] == 1)
        row_cover_next = state.row_cover.at[row].set(True)
        col_cover_next = state.col_cover.at[col].set(False)
        return StateStruct(state.M, state.mask_mx, row_cover_next, col_cover_next)

    def aug_path(state: StateStruct, row: int, col: int) -> StateStruct:
        # Step 5 (and col_cover of Step 3)
        AugPathStateStruct = namedtuple("AugPathStateStruct", ["aug_path", "mask_mx", "row", "col"])
        aug_path = jnp.zeros_like(state.mask_mx).at[row, col].set(1)

        def cond_func(aug_path_state: AugPathStateStruct) -> bool:
            return jnp.sum(aug_path_state.mask_mx[:, aug_path_state.col] == 1) > 0

        def body_func(aug_path_state: AugPathStateStruct) -> AugPathStateStruct:
            row = jnp.argmax(aug_path_state.mask_mx[:, aug_path_state.col] == 1) 
            col = jnp.argmax(aug_path_state.mask_mx[row] == 2)
            aug_path_next = aug_path_state.aug_path.at[row, aug_path_state.col].set(1)
            aug_path_next = aug_path_next.at[row, col].set(1)
            return AugPathStateStruct(aug_path_next, aug_path_state.mask_mx, row, col)

        aug_path_state = AugPathStateStruct(aug_path, state.mask_mx, row, col)
        aug_path_state = bounded_while_loop(cond_func, body_func, aug_path_state, max_steps=n)
        mask_mx_next = state.mask_mx - aug_path_state.aug_path
        mask_mx_next = jnp.where(mask_mx_next == 2, 0, mask_mx_next)
        col_cover = jnp.sum(mask_mx_next == 1, axis=0).astype(bool)
        return StateStruct(state.M, mask_mx_next, jnp.zeros((n,), dtype=bool), col_cover)

    def find_hard_assignment(state: StateStruct) -> StateStruct:
        # Step 4 (Steps 5 & 6 nested)
        def cond_func(state: StateStruct) -> bool:
            # condition part of Step 3
            return jnp.sum(state.mask_mx == 1) < n

        def body_func(state: StateStruct) -> StateStruct:
            cover_mask = state.row_cover.reshape(-1, 1) | state.col_cover.reshape(1, -1)
            uncovered_zero_mask = (state.M == 0) & (~cover_mask)

            def prime_or_aug_path(state: StateStruct, uncovered_zero_mask: Array) -> StateStruct:
                flattened_idx = jnp.argmax(uncovered_zero_mask)
                row, col = flattened_idx // n, flattened_idx % n
                mask_mx_updated = state.mask_mx.at[row, col].set(2)
                next_state = StateStruct(state.M, mask_mx_updated, state.row_cover, state.col_cover)
                next_state = lax.cond(
                    jnp.sum(mask_mx_updated[row] == 1) >= 1, adjust_cover, aug_path, next_state, row, col)
                return next_state

            next_state = lax.cond(
                jnp.sum(uncovered_zero_mask) == 0,
                lambda state, _: shift_zeros(state),
                lambda state, uncovered_zero_mask: prime_or_aug_path(state, uncovered_zero_mask),
                state,
                uncovered_zero_mask)
            return next_state

        return bounded_while_loop(cond_func, body_func, state, max_steps=(n**4)).mask_mx

    state = StateStruct(
        final_state.M,
        final_state.mask_mx,
        jnp.zeros((n,), dtype=bool),
        jnp.sum(final_state.mask_mx == 1, axis=0).astype(bool))
    return find_hard_assignment(state)


if __name__ == "__main__":

    # TODO: test other cases
    #C = jnp.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
    #C = jnp.array([[7, 6, 2, 9, 2],
    #               [6, 2, 1, 3, 9],
    #               [5, 6, 8, 9, 5],
    #               [6, 8, 5, 8, 6],
    #               [9, 5, 6, 4, 7]])
    C = jnp.array([[62,75,80,93,95,97],
                   [75,80,82,85,71,97],
                   [80,75,81,98,90,97],
                   [78,82,84,80,50,98],
                   [90,85,85,80,85,99],
                   [65,75,80,75,68,96]])
    #C = jnp.array([[1,2,3,4],[2,4,6,8],[3,6,9,12],[4,8,12,16]])
    #C = jnp.array([[1,2,3],[3,3,3],[3,3,2]])
    #C = jnp.array([[7,4,3],[3,1,2],[3,0,0]])
    #C = jnp.array([[-1,-2,-3],[-3,-3,-3],[-3,-3,-2]])
    n = C.shape[0]
    assignment = munkres(n, C)

    print("C: \n", C)
    print("assignment: \n", assignment)



