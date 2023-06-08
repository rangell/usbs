import jax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
from typing import Any, Callable, Tuple

from IPython import embed


def load_and_process_qap(fname: str) -> Tuple[Array, Array]:
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


def load_and_process_tsp(fname: str) -> Tuple[Array, Array]:
    # used documentation to implement: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp95.pdf
    spec_vars = {}
    with open(fname, "r") as f:
        for line in f:
            splitline = [ss.strip() for ss in line.split(":")]
            if len(splitline) == 2: # specification part
                spec_vars[splitline[0]] = splitline[1]
            elif len(splitline) == 1: # info part
                if splitline[0] == "NODE_COORD_SECTION":
                    coords = []
                    while True:
                        line = next(f).strip()
                        if line == "EOF":
                            assert len(coords) == int(spec_vars['DIMENSION'])
                            break
                        _, _x, _y = tuple(line.split())
                        coords.append([_x, _y])
                    break # break out of initial for loop
                elif splitline[0] == "EDGE_WEIGHT_SECTION":
                    # TODO: handle EDGE_WEIGHT_SECTION case, end is either "DISPLAY_DATA_SECTION" or "EOF"
                    edge_weight_str = ""
                    while True:
                        line = next(f).strip()
                        if line == "EOF" or line == "DISPLAY_DATA_SECTION":
                            break
                        edge_weight_str += " " + line
                    flat_edge_weights = jnp.array([int(v) for v in edge_weight_str.split()])
                    if spec_vars["EDGE_WEIGHT_FORMAT"] == "FULL_MATRIX":
                        pass
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] == "UPPER_ROW":
                        pass
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] == "LOWER_ROW":
                        pass
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] == "UPPER_DIAG_ROW":
                        pass
                    elif spec_vars["EDGE_WEIGHT_FORMAT"] == "UPPER_DIAG_ROW":
                        pass
                    embed()
                    exit()
                else:
                    pass
            else:
                raise ValueError("Something went wrong when reading file.")
        embed()
        exit()
            
    # TODO: build data

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


def get_all_problem_data(C: BCOO) -> Tuple[BCOO, Array, Array, Array]:
    n = C.shape[0]
    l = int(jnp.sqrt(n - 1))
    i = 0  # running constraint-matrix index

    # initialize with first constraint: X(0,0) = 1
    A_indices = [[0, 0, 0]]
    A_data = [1.0]
    b = [1.0]
    i += 1

    # constraint: diag(Y) = vec(P)
    for j in range(1, n):
        A_indices.append([i, j, 0])
        A_indices.append([i, 0, j])
        A_indices.append([i, j, j])
        A_data += [-0.5, -0.5, 1.0]
        b.append(0.0)
        i += 1

    # constraint: P1 = 1
    for j1 in range(l):
        for j2 in range(l):
            A_indices.append([i, j1*l + j2, 0])
            A_indices.append([i, 0, j1*l + j2])
            A_data += [0.5, 0.5]
        b.append(1.0)
        i += 1

    # constraint: 1'P = 1'
    for j1 in range(l):
        for j2 in range(l):
            A_indices.append([i, j1 + j2*l, 0])
            A_indices.append([i, 0, j1 + j2*l])
            A_data += [0.5, 0.5]
        b.append(1.0)
        i += 1

    # constraint: tr_1(Y) = I
    for j1 in range(l):
        for j2 in range(j1, l):
            for diag_idx in range(l):
                if j1 == j2:
                    A_indices.append([i, j1 + diag_idx*l + 1, j1 + diag_idx*l + 1])
                    A_data += [1.0]
                else:
                    A_indices.append([i, j1 + diag_idx*l + 1, j2 + diag_idx*l + 1])
                    A_indices.append([i, j2 + diag_idx*l + 1, j1 + diag_idx*l + 1])
                    A_data += [0.5, 0.5]
            if j1 == j2:
                b.append(1.0)
            else:
                b.append(0.0)
            i += 1

    # constraint: tr_2(Y) = I
    for j1 in range(l):
        for j2 in range(j1, l):
            for diag_idx in range(l):
                if j1 == j2:
                    A_indices.append([i, j1*l + diag_idx + 1, j1*l + diag_idx + 1])
                    A_data += [1.0]
                else:
                    A_indices.append([i, j1*l + diag_idx + 1, j2*l + diag_idx + 1])
                    A_indices.append([i, j2*l + diag_idx + 1, j1*l + diag_idx + 1])
                    A_data += [0.5, 0.5]
            if j1 == j2:
                b.append(1.0)
            else:
                b.append(0.0)
            i += 1

    # constraint: objective-relevant entries of Y >= 0
    added_dims = 0
    for j in range(C.nse):
        coord_a, coord_b = C.indices[j][0], C.indices[j][1]
        if coord_a < coord_b:
            A_indices.append([i, coord_a, coord_b])
            A_indices.append([i, coord_b, coord_a])
            A_indices.append([i, n + added_dims, n + added_dims])
            A_data += [0.5, 0.5, -1.0]
            b.append(0.0)
            added_dims += 1
            i += 1
        elif coord_a == coord_b:
            A_indices.append([i, coord_a, coord_a])
            A_indices.append([i, n + added_dims, n + added_dims])
            A_data += [1.0, -1.0]
            b.append(0.0)
            added_dims += 1
            i += 1

    # build final data structures
    C = BCOO((C.data, C.indices), shape=(n + added_dims, n + added_dims))
    A_indices = jnp.array(A_indices)
    A_data = jnp.array(A_data)
    b = jnp.array(b)

    return C, A_indices, A_data, b


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    #DATAFILE = "data/qap/qapdata/chr12a.dat"
    #DATAFILE = "data/qap/tspdata/ulysses16.tsp"
    DATAFILE = "data/qap/tspdata/dantzig42.tsp"

    # TODO: write load TSP
    if DATAFILE.split(".")[-1] == "dat":
        C = load_and_process_qap(DATAFILE)
    elif DATAFILE.split(".")[-1] == "tsp":
        C = load_and_process_tsp(DATAFILE)
    else:
        raise ValueError("Invalid data file type.")

    C, A_indices, A_data, b = get_all_problem_data(C)

    embed()
    exit()