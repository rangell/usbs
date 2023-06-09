import jax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import numpy as np
import scipy  # type: ignore
from scipy.spatial.distance import pdist, squareform  # type: ignore
from typing import Any, Callable, Tuple

from solver.cgal import cgal
from solver.specbm import specbm
from solver.utils import reconstruct_from_sketch

from IPython import embed


def load_and_process_qap(fname: str) -> Tuple[Array, Array]:
    with open(fname, "r") as f:
        datastr = f.read()
        str_n, str_D, str_W, _ = tuple(datastr.split("\n\n"))
        n = int(str_n)
        D = jnp.array([float(v) for v in str_D.split()]).reshape(n, n)
        W = jnp.array([float(v) for v in str_W.split()]).reshape(n, n)

    # by convention, W should be sparser than D -- this doesn't really matter
    if jnp.count_nonzero(D) < jnp.count_nonzero(W):
        D, W = W, D

    # return expanded and padded kronecker product
    return n, build_objective_matrix(D, W)


def load_and_process_tsp(fname: str) -> Tuple[Array, Array]:
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
            
    # construct sparse symmetric canonical tour
    W_indices = []
    W_data = []
    for i in range(n):
        W_indices.append([i, (i + 1) % n])
        W_indices.append([(i + 1) % n, i])
        W_data += [0.5, 0.5]
    W_indices = jnp.array(W_indices)
    W_data = jnp.array(W_data)
    W = BCOO((W_data, W_indices), shape=(n, n)).todense()

    # return expanded and padded kronecker product
    return n, build_objective_matrix(D, W)


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

    DATAFILE = "data/qap/qapdata/chr12a.dat"
    #DATAFILE = "data/qap/tspdata/ulysses16.tsp"
    #DATAFILE = "data/qap/tspdata/dantzig42.tsp"
    #DATAFILE = "data/qap/tspdata/bayg29.tsp"
    #DATAFILE = "data/qap/tspdata/bays29.tsp"
    #DATAFILE = "data/qap/tspdata/att48.tsp"

    # TODO: write load TSP
    if DATAFILE.split(".")[-1] == "dat":
        l, C = load_and_process_qap(DATAFILE)
    elif DATAFILE.split(".")[-1] == "tsp":
        l, C = load_and_process_tsp(DATAFILE)
    else:
        raise ValueError("Invalid data file type.")

    C, A_indices, A_data, b = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    SCALE_X = 1.0 / float(n)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = jnp.zeros((m,))
    SCALE_A = SCALE_A.at[A_indices[:,0]].add(A_data**2)
    SCALE_A = 1.0 / jnp.sqrt(SCALE_A)

    scaled_C = BCOO((C.data * SCALE_C, C.indices), shape=C.shape)
    b = b * SCALE_X * SCALE_A

    # TODO: rescale A_data by SCALE_A
    scaled_A_data = A_data * SCALE_A.at[A_indices[:,0]].get()

    SCALE_SCALED_A = jnp.zeros((m,))
    SCALE_SCALED_A = SCALE_SCALED_A.at[A_indices[:,0]].add(scaled_A_data**2)
    SCALE_SCALED_A = 1.0 / jnp.sqrt(SCALE_SCALED_A)

    embed()
    exit()

    X = jnp.zeros((n, n))
    P = None
    Omega = None
    y = jnp.zeros((m,))
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    trace_ub = 1.0 * float(n) * SCALE_X

    X, P, y, z, primal_obj, tr_X = cgal(
        X=X,
        P=P,
        y=y,
        z=z,
        primal_obj=primal_obj,
        tr_X=tr_X,
        n=n,
        m=m,
        trace_ub=trace_ub,
        C=scaled_C,
        A_data=A_data,
        A_indices=A_indices,
        b=b,
        Omega=Omega,
        beta0=1.0,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        eps=1e-3,  # hparams.eps,
        max_iters=50,  # hparams.max_iters,
        line_search=False,  # hparams.cgal_line_search,
        lanczos_inner_iterations=min(n, 32),
        lanczos_max_restarts=10,  # hparams.lanczos_max_restarts,
        subprob_tol=1e-7,
        callback_fn=None)

    embed()
    exit()