from functools import partial
import jax
from jax import lax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import numba as nb
import numpy as np
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform  # type: ignore
from typing import Any, Tuple

from solver.utils import apply_A_operator_batched
from utils.common import (SDPState,
                          scale_sdp_state,
                          unscale_sdp_state,
                          reconstruct_from_sketch,
                          apply_A_operator_mx)
from utils.munkres import munkres

from IPython import embed


def load_and_process_qap(fname: str, num_drop: int = 0) -> Tuple[Array, Array]:
    with open(fname, "r") as f:
        datastr = f.read()
        split_index = datastr.find("\n\n")
        str_n, str_data = datastr[:split_index], datastr[split_index+2:]
        n = int(str_n)
        M = jnp.array([float(v) for v in str_data.split()]).reshape(2*n, n)
        D = M[:n]
        W = M[n:]

    # by convention, W should be sparser than D -- this doesn't really matter
    if jnp.count_nonzero(D) < jnp.count_nonzero(W):
        D, W = W, D

    n_out = n - num_drop
    D = D[:n_out, :n_out]
    W = W[:n_out, :n_out]

    # return expanded and padded kronecker product
    return n_out, D, W, build_objective_matrix(D, W)


def load_and_process_tsp(fname: str, num_drop: int = 0) -> Tuple[Array, Array]:
    # used documentation to implement: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
    spec_vars = {}
    with open(fname, "r") as f:
        for line in f:
            splitline = [ss.strip() for ss in line.split(":")]
            if len(splitline) == 2: # specification part
                spec_vars[splitline[0]] = splitline[1]
            elif len(splitline) == 1: # info part
                n = int(spec_vars["DIMENSION"])
                D = jnp.zeros((n, n))
                filled_D = False
                coords = None
                if splitline[0] == "NODE_COORD_SECTION":
                    coords = []
                    while True:
                        line = next(f).strip()
                        if line == "EOF":
                            assert len(coords) == n
                            break
                        _, _x, _y = tuple(line.split())
                        coords.append([float(_x), float(_y)])
                    break  # break out of initial for loop, we have what we need.
                elif splitline[0] == "EDGE_WEIGHT_SECTION":
                    edge_dist_str = ""
                    while True:
                        line = next(f).strip()
                        if line == "EOF" or line == "DISPLAY_DATA_SECTION":
                            break
                        edge_dist_str += " " + line
                    flat_edge_dists = jnp.array([int(v) for v in edge_dist_str.split()])
                    if spec_vars["EDGE_WEIGHT_FORMAT"] == "FULL_MATRIX":
                        indices = np.where(D == 0)
                        D = D.at[indices].set(flat_edge_dists)
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] in ["UPPER_ROW", "LOWER_COL"]:
                        indices = jnp.triu_indices(n, k=1)
                        D = D.at[indices].set(flat_edge_dists)
                        D = D + D.T
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] in ["LOWER_ROW", "UPPER_COL"]:
                        raise NotImplementedError("edge weight format type not implemented")  # no examples for this case?
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] in ["UPPER_DIAG_ROW", "LOWER_DIAG_COL"]:
                        indices = jnp.triu_indices(n, k=0)
                        D = D.at[indices].set(flat_edge_dists)
                        D = D + D.T
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] in ["LOWER_DIAG_ROW", "UPPER_DIAG_COL"]:
                        indices = jnp.tril_indices(n, k=0)
                        D = D.at[indices].set(flat_edge_dists)
                        D = D + D.T
                    else:
                        raise ValueError("Unsupported EDGE_WEIGHT_FORMAT.")
                    filled_D = True
                    break  # break out of initial for loop, we have what we need.
                else:
                    pass
            else:
                raise ValueError("Something went wrong when reading file.")

        if not filled_D:
            assert coords is not None
            coords = np.array(coords)
            if spec_vars["EDGE_WEIGHT_TYPE"] == "GEO":
                def geo_dist(p1, p2):
                    PI = 3.141592
                    RRR = 6378.388

                    # compute lat-long for each point in radians
                    lat1 = PI * ((deg := np.round(p1[0])) + 5.0 * (p1[0] - deg) / 3.0 ) / 180.0
                    long1 = PI * ((deg := np.round(p1[1])) + 5.0 * (p1[1] - deg) / 3.0 ) / 180.0
                    lat2 = PI * ((deg := np.round(p2[0])) + 5.0 * (p2[0] - deg) / 3.0 ) / 180.0
                    long2 = PI * ((deg := np.round(p2[1])) + 5.0 * (p2[1] - deg) / 3.0 ) / 180.0

                    # compute distance in kilometers
                    q1 = np.cos(long1 - long2)
                    q2 = np.cos(lat1 - lat2)
                    q3 = np.cos(lat1 + lat2)
                    return int(RRR * np.arccos(0.5*(q2 * (1.0 + q1) - q3 * (1.0 - q1))) + 1.0)
                D = squareform(pdist(coords, geo_dist))
            elif spec_vars["EDGE_WEIGHT_TYPE"] == "ATT":
                def att_dist(p1, p2):
                    xd = p1[0] - p2[0]
                    yd = p1[1] - p2[1]
                    return np.ceil(np.sqrt((xd**2 + yd**2) / 10.0))
                D = squareform(pdist(coords, att_dist))
            elif spec_vars["EDGE_WEIGHT_TYPE"] == "EUC_2D":
                D = np.round(squareform(pdist(coords, 'euclidean')))
            elif spec_vars["EDGE_WEIGHT_TYPE"] == "CEIL_2D":
                D = np.ceil(squareform(pdist(coords, 'euclidean')))
            else:
                raise ValueError("Unsupported EDGE_WEIGHT_TYPE.")
            D = jnp.array(D)
            

    n_out = n - num_drop
    D = D[:n_out, :n_out]

    # construct sparse symmetric canonical tour
    W_indices = []
    W_data = []
    for i in range(n_out):
        W_indices.append([i, (i + 1) % n_out])
        W_indices.append([(i + 1) % n_out, i])
        W_data += [0.5, 0.5]
    W_indices = jnp.array(W_indices)
    W_data = jnp.array(W_data)
    W = BCOO((W_data, W_indices), shape=(n_out, n_out)).todense()

    # return expanded and padded kronecker product
    return n_out, D, W, build_objective_matrix(D, W)


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


def get_all_problem_data(C: BCOO) -> Tuple[BCOO, Array, Array, Array]:
    n = C.shape[0]
    l = int(jnp.sqrt(n - 1))

    # initialize with first constraint: X(0,0) = 1
    A_indices = jnp.array([[0, 0, 0]])
    A_data = jnp.array([1.0])
    b = jnp.array([1.0])
    b_ineq_mask = jnp.array([0.0])

    # TODO: change this from "P" to some other variable depending on paper
    # constraint: diag(Y) = vec(P)
    # equivalent to the following:
    #   for j in range(1, n):
    #       _A_indices.append([_i, j, 0])
    #       _A_indices.append([_i, 0, j])
    #       _A_indices.append([_i, j, j])
    #       _A_data += [-0.5, -0.5, 1.0]
    #       _b.append(0.0)
    #       _b_ineq_mask.append(0.0)
    #       _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, n-1)[:, None], (1, 3)).reshape(-1,)
    coord_a = (jnp.tile(jnp.array([[1, 0, 1]]), (n-1, 1)) * jnp.arange(1, n)[:, None]).reshape(-1,)
    coord_b = (jnp.tile(jnp.array([[0, 1, 1]]), (n-1, 1)) * jnp.arange(1, n)[:, None]).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_data = jnp.concatenate(
        [A_data, jnp.tile(jnp.array([[-1.0, -1.0, 2.0]]), (n-1, 1)).reshape(-1,)], axis=0)
    b = jnp.concatenate([b, jnp.zeros((n-1,))], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((n-1,))], axis=0)

    # TODO: change this from "P" to some other variable depending on paper
    # constraint: P1 = 1
    # equivalent to the following:
    #   for j1 in range(l):
    #       for j2 in range(l):
    #           _A_indices.append([_i, j1*l + j2 + 1, 0])
    #           _A_indices.append([_i, 0, j1*l + j2 + 1])
    #           _A_data += [0.5, 0.5]
    #       _b.append(1.0)
    #       _b_ineq_mask.append(0.0)
    #       _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, l)[:, None], (1, 2*l)).reshape(-1,)
    coord_a = (jnp.tile(jnp.array([[1, 0]]), (n-1, 1))
               * jnp.arange(1, n).reshape(l, l).flatten()[:, None]).reshape(-1,)
    coord_b = (jnp.tile(jnp.array([[0, 1]]), (n-1, 1))
               * jnp.arange(1, n).reshape(l, l).flatten()[:, None]).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_data = jnp.concatenate(
        [A_data, jnp.tile(jnp.array([[1.0, 1.0]]), (n-1, 1)).reshape(-1,)], axis=0)
    b = jnp.concatenate([b, 2.0*jnp.ones((l,))], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((l,))], axis=0)


    ## TODO: change this from "P" to some other variable depending on paper
    # constraint: 1'P = 1'
    # equivalent to the following:
    #   for j1 in range(l):
    #       for j2 in range(l):
    #           _A_indices.append([_i, j1 + j2*l + 1, 0])
    #           _A_indices.append([_i, 0, j1 + j2*l + 1])
    #           _A_data += [0.5, 0.5]
    #       _b.append(1.0)
    #       _b_ineq_mask.append(0.0)
    #       _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, l)[:, None], (1, 2*l)).reshape(-1,)
    coord_a = (jnp.tile(jnp.array([[1, 0]]), (n-1, 1))
               * jnp.arange(1, n).reshape(l, l).T.flatten()[:, None]).reshape(-1,)
    coord_b = (jnp.tile(jnp.array([[0, 1]]), (n-1, 1))
               * jnp.arange(1, n).reshape(l, l).T.flatten()[:, None]).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_data = jnp.concatenate(
        [A_data, jnp.tile(jnp.array([[1.0, 1.0]]), (n-1, 1)).reshape(-1,)], axis=0)
    b = jnp.concatenate([b, 2.0*jnp.ones((l,))], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((l,))], axis=0)

    ## constraint: tr_1(Y) = I
    ## equivalent to the following:
    #   for j1 in range(l):
    #       for j2 in range(l):
    #           for diag_idx in range(l):
    #               _A_indices.append([_i, j1 + diag_idx*l + 1, j2 + diag_idx*l + 1])
    #               _A_data += [1.0]
    #           if j1 == j2:
    #               _b.append(1.0)
    #           else:
    #               _b.append(0.0)
    #           _b_ineq_mask.append(0.0)
    #           _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, n-1)[:, None], (1, l)).reshape(-1,)
    coord_a = 1 + jnp.tile(
        jnp.arange(l)[:, None, None]
        + l * jnp.arange(l)[None, None, :], (1, l, 1)).reshape(-1,)
    coord_b = 1 + jnp.tile(
        jnp.arange(l)[None, :, None]
        + l * jnp.arange(l)[None, None, :], (l, 1, 1)).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_b, coord_a]).T], axis=0)
    A_data = jnp.concatenate([A_data, jnp.ones_like(coord_a), jnp.ones_like(coord_a)], axis=0)
    b = jnp.concatenate([b, 2.0*(coord_a == coord_b).reshape(l**2, l).T[0].astype(float)], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((n-1,))], axis=0)

    # constraint: tr_2(Y) = I
    # equivalent to the following:
    #   for j1 in range(l):
    #       for j2 in range(l):
    #           for diag_idx in range(l):
    #               _A_indices.append([_i, j1*l + diag_idx + 1, j2*l + diag_idx + 1])
    #               _A_data += [1.0]
    #           if j1 == j2:
    #               _b.append(1.0)
    #           else:
    #               _b.append(0.0)
    #           _b_ineq_mask.append(0.0)
    #           _i += 1
    constraint_indices = b.shape[0] + jnp.tile(jnp.arange(0, n-1)[:, None], (1, l)).reshape(-1,)
    coord_a = 1 + jnp.tile(
        l * jnp.arange(l)[:, None, None]
        + jnp.arange(l)[None, None, :], (1, l, 1)).reshape(-1,)
    coord_b = 1 + jnp.tile(
        l * jnp.arange(l)[None, :, None]
        + jnp.arange(l)[None, None, :], (l, 1, 1)).reshape(-1,)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_a, coord_b]).T], axis=0)
    A_indices = jnp.concatenate(
        [A_indices, jnp.vstack([constraint_indices, coord_b, coord_a]).T], axis=0)
    A_data = jnp.concatenate([A_data, jnp.ones_like(coord_a), jnp.ones_like(coord_a)], axis=0)
    b = jnp.concatenate([b, 2.0*(coord_a == coord_b).reshape(l**2, l).T[0].astype(float)], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.zeros((n-1,))], axis=0)

    # constraint: objective-relevant entries of Y >= 0, written as -Y <= 0
    triu_indices_mask = (C.indices[:, 0] <= C.indices[:, 1])
    constraint_indices = b.shape[0] + jnp.arange(jnp.sum(triu_indices_mask))
    constraint_triples = jnp.concatenate(
        [constraint_indices[:, None], C.indices[triu_indices_mask]], axis=1)
    constraint_triples = jnp.concatenate(
        [constraint_triples, constraint_triples[:, [0, 2, 1]]], axis=0)
    A_indices = jnp.concatenate([A_indices, constraint_triples], axis=0)
    A_data = jnp.concatenate([A_data, jnp.full((constraint_triples.shape[0],), -1.0)], axis=0)
    b = jnp.concatenate([b, jnp.full((constraint_indices.shape[0],), 0.0)], axis=0)
    b_ineq_mask = jnp.concatenate([b_ineq_mask, jnp.full((constraint_indices.shape[0],), 1.0)], axis=0)

    return A_data, A_indices, b, b_ineq_mask


def initialize_state(C: BCOO, sketch_dim: int) -> SDPState:
    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]
    l = int(jnp.sqrt(n - 1))

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False)[0])
    SCALE_A /= norm_A

    if sketch_dim == -1:
        X = jnp.zeros((n, n))
        Omega = None
        P = None
    elif sketch_dim > 0:
        X = None
        Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, sketch_dim))
        P = jnp.zeros_like(Omega)
    else:
        raise ValueError("Invalid value for sketch_dim")

    y = jnp.zeros((m,))
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


@nb.njit
def _fill_constraint_index_map(old_A_indices, new_A_indices, constraint_index_map) -> None:
    old_idx = 0
    for curr_idx in range(new_A_indices.shape[0]):
        old_row = old_A_indices[old_idx]
        curr_row = new_A_indices[curr_idx]
        if curr_row[1] == old_row[1] and curr_row[2] == old_row[2]:
            constraint_index_map[old_row[0]] = curr_row[0]
            old_idx += 1


def get_implicit_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    assert sketch_dim == -1 or sketch_dim == old_sdp_state.Omega.shape[1]
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    old_l = int(jnp.sqrt(old_sdp_state.C.shape[0] - 1))
    old_m = old_sdp_state.b.shape[0]
    l = int(jnp.sqrt(C.shape[0] - 1))
    num_drop = l - old_l
    assert sketch_dim in [-1, l]
    
    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    index_map = lambda a : a + num_drop * jnp.clip(((a - 1) // (l - num_drop)), a_min=0)

    X = old_sdp_state.X
    Omega = old_sdp_state.Omega
    P = old_sdp_state.P
    if old_sdp_state.X is not None:
        X = BCOO.fromdense(old_sdp_state.X)
        X = BCOO((X.data, jax.vmap(index_map)(X.indices)), shape=(n, n)).todense()
    if old_sdp_state.P is not None:
        Omega = jax.random.normal(jax.random.PRNGKey(n), shape=(n, l)).at[jax.vmap(index_map)(
            jnp.arange(old_sdp_state.Omega.shape[0]))].set(old_sdp_state.Omega)
        P = jnp.zeros_like(Omega).at[jax.vmap(index_map)(
            jnp.arange(old_sdp_state.P.shape[0]))].set(old_sdp_state.P)
    
    old_A_indices = old_sdp_state.A_indices.at[:, 1:].set(
        jax.vmap(index_map)(old_sdp_state.A_indices[:, 1:]))
    constraint_index_map = np.empty((old_m,), dtype=int)
    _fill_constraint_index_map(
        np.asarray(old_A_indices), np.asarray(A_indices), constraint_index_map)

    y = jnp.zeros((m,)).at[constraint_index_map].set(old_sdp_state.y)
    z = jnp.zeros((m,)).at[constraint_index_map].set(old_sdp_state.z)

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False)[0])
    SCALE_A /= norm_A

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=old_sdp_state.tr_X,
        primal_obj=old_sdp_state.primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state

def get_explicit_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    old_l = int(jnp.sqrt(old_sdp_state.C.shape[0] - 1))
    old_m = old_sdp_state.b.shape[0]
    l = int(jnp.sqrt(C.shape[0] - 1))
    num_drop = l - old_l
    assert sketch_dim in [-1, l]
    
    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    index_map = lambda a : a + num_drop * jnp.clip(((a - 1) // (l - num_drop)), a_min=0)

    old_A_indices = old_sdp_state.A_indices.at[:, 1:].set(
        jax.vmap(index_map)(old_sdp_state.A_indices[:, 1:]))
    constraint_index_map = np.empty((old_m,), dtype=int)
    _fill_constraint_index_map(
        np.asarray(old_A_indices), np.asarray(A_indices), constraint_index_map)

    X = old_sdp_state.X
    Omega = old_sdp_state.Omega
    P = old_sdp_state.P
    if old_sdp_state.X is not None:
        X = BCOO.fromdense(old_sdp_state.X)
        X = BCOO((X.data, jax.vmap(index_map)(X.indices)), shape=(n, n)).todense()
        z = apply_A_operator_mx(n, m, A_data, A_indices, X) 
        tr_X = jnp.trace(X)
        primal_obj = jnp.trace(C @ X)
    if old_sdp_state.P is not None:
        Omega = jax.random.normal(jax.random.PRNGKey(n), shape=(n, sketch_dim))
        E, Lambda = reconstruct_from_sketch(old_sdp_state.Omega, old_sdp_state.P)
        tr_offset = (old_sdp_state.tr_X - jnp.sum(Lambda)) / Lambda.shape[0]
        Lambda_tr_correct = Lambda + tr_offset
        E = jnp.zeros_like(Omega).at[jax.vmap(index_map)(jnp.arange(E.shape[0]))].set(E)
        sqrt_X_hat = E * jnp.sqrt(Lambda_tr_correct)[None, :]
        P = sqrt_X_hat @ (sqrt_X_hat.T @ Omega)
        z = apply_A_operator_batched(m, A_data, A_indices, sqrt_X_hat)
        tr_X = jnp.sum(Lambda_tr_correct)
        primal_obj = jnp.trace(sqrt_X_hat.T @ (C @ sqrt_X_hat))

    y = jnp.zeros((m,)).at[constraint_index_map].set(old_sdp_state.y)

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False)[0])
    SCALE_A /= norm_A

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


def get_dual_only_warm_start_state(old_sdp_state: SDPState, C: BCOO, sketch_dim: int) -> SDPState:
    old_sdp_state = unscale_sdp_state(old_sdp_state)

    A_data, A_indices, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    old_l = int(jnp.sqrt(old_sdp_state.C.shape[0] - 1))
    old_m = old_sdp_state.b.shape[0]
    l = int(jnp.sqrt(C.shape[0] - 1))
    num_drop = l - old_l
    assert sketch_dim in [-1, l]
    
    index_map = lambda a : a + num_drop * jnp.clip(((a - 1) // (l - num_drop)), a_min=0)

    old_A_indices = old_sdp_state.A_indices.at[:, 1:].set(
        jax.vmap(index_map)(old_sdp_state.A_indices[:, 1:]))
    constraint_index_map = np.empty((old_m,), dtype=int)
    _fill_constraint_index_map(
        np.asarray(old_A_indices), np.asarray(A_indices), constraint_index_map)

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = 1.0 / jnp.sqrt(jnp.zeros((m,)).at[A_indices[:,0]].add(A_data**2))
    A_tensor = BCOO((A_data, A_indices), shape=(m, n, n))
    A_matrix = SCALE_A[:, None] * A_tensor.reshape(m, n**2)
    A_matrix = coo_matrix(
        (A_matrix.data, (A_matrix.indices[:,0], A_matrix.indices[:,1])), shape=A_matrix.shape)
    norm_A = jnp.sqrt(eigsh(A_matrix @ A_matrix.T, k=1, which="LM", return_eigenvectors=False)[0])
    SCALE_A /= norm_A

    if sketch_dim == -1:
        X = jnp.zeros((n, n))
        Omega = None
        P = None
    elif sketch_dim > 0:
        X = None
        Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, sketch_dim))
        P = jnp.zeros_like(Omega)
    else:
        raise ValueError("Invalid value for sketch_dim")

    y = jnp.zeros((m,)).at[constraint_index_map].set(old_sdp_state.y)
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    sdp_state = SDPState(
        C=C,
        A_indices=A_indices,
        A_data=A_data,
        b=b,
        b_ineq_mask=b_ineq_mask,
        X=X,
        P=P,
        Omega=Omega,
        y=y,
        z=z,
        tr_X=tr_X,
        primal_obj=primal_obj,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A)

    sdp_state = scale_sdp_state(sdp_state)
    return sdp_state


@partial(jax.jit, static_argnames=["callback_static_args"])
def qap_round(
    P: Array,
    Omega: Array,
    callback_static_args: bytes,
    callback_nonstatic_args: Any
) -> float:
    l = pickle.loads(callback_static_args)["l"]
    D = callback_nonstatic_args["D"]
    W = callback_nonstatic_args["W"]
    E, _ = reconstruct_from_sketch(Omega, P)
    def body_func(i: int, best_assign_obj: float) -> float:
        cost_mx = E[1:, i].reshape(l, l)
        cost_mx = jnp.max(cost_mx) - cost_mx
        perm_mx = munkres(l, cost_mx)
        best_assign_obj = jnp.clip(jnp.trace(W @ perm_mx @ D @ perm_mx.T), a_max=best_assign_obj)
        # try negative too
        cost_mx = -E[1:, i].reshape(l, l)
        cost_mx = jnp.max(cost_mx) - cost_mx
        perm_mx = munkres(l, cost_mx)
        best_assign_obj = jnp.clip(jnp.trace(W @ perm_mx @ D @ perm_mx.T), a_max=best_assign_obj)
        return best_assign_obj
    best_assign_obj = lax.fori_loop(0, l, body_func, jnp.inf)
    return best_assign_obj