import argparse
import jax
from jax import lax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import jax.numpy as jnp
import json
import numpy as np
from scipy.spatial.distance import pdist, squareform  # type: ignore
from typing import Any, Callable, Tuple

from scripts.munkres import munkres
from solver.specbm import specbm
from solver.utils import reconstruct_from_sketch

from IPython import embed


def load_and_process_qap(fname: str, num_drop: int = 0) -> Tuple[int, Array, Array, Array]:
    with open(fname, "r") as f:
        datastr = f.read()
        str_n, str_D, str_W, _ = tuple(datastr.split("\n\n"))
        n = int(str_n)
        D = jnp.array([float(v) for v in str_D.split()]).reshape(n, n)
        W = jnp.array([float(v) for v in str_W.split()]).reshape(n, n)

    # for warm_starting
    n = n - num_drop
    D = D[:n, :n]
    W = W[:n, :n]

    # by convention, W should be sparser than D -- this doesn't really matter
    if jnp.count_nonzero(D) < jnp.count_nonzero(W):
        D, W = W, D

    # return expanded and padded kronecker product
    return n, D, W, build_objective_matrix(D, W)


def load_and_process_tsp(fname: str, num_drop: int = 0) -> Tuple[int, Array, Array, Array]:
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

    # for warm_starting
    n = n - num_drop
    D = D[:n, :n]
    W = W[:n, :n]

    # return expanded and padded kronecker product
    return n, D, W, build_objective_matrix(D, W)


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
    b_ineq_mask = [0.0]
    i += 1

    # constraint: diag(Y) = vec(P)
    for j in range(1, n):
        A_indices.append([i, j, 0])
        A_indices.append([i, 0, j])
        A_indices.append([i, j, j])
        A_data += [-0.5, -0.5, 1.0]
        b.append(0.0)
        b_ineq_mask.append(0.0)
        i += 1

    # constraint: P1 = 1
    for j1 in range(l):
        for j2 in range(l):
            A_indices.append([i, j1*l + j2 + 1, 0])
            A_indices.append([i, 0, j1*l + j2 + 1])
            A_data += [0.5, 0.5]
        b.append(1.0)
        b_ineq_mask.append(0.0)
        i += 1

    # constraint: 1'P = 1'
    for j1 in range(l):
        for j2 in range(l):
            A_indices.append([i, j1 + j2*l + 1, 0])
            A_indices.append([i, 0, j1 + j2*l + 1])
            A_data += [0.5, 0.5]
        b.append(1.0)
        b_ineq_mask.append(0.0)
        i += 1

    ## constraint: tr_1(Y) = I
    for j1 in range(l):
        for j2 in range(l):
            for diag_idx in range(l):
                A_indices.append([i, j1 + diag_idx*l + 1, j2 + diag_idx*l + 1])
                A_data += [1.0]
            if j1 == j2:
                b.append(1.0)
            else:
                b.append(0.0)
            b_ineq_mask.append(0.0)
            i += 1

    ## constraint: tr_2(Y) = I
    for j1 in range(l):
        for j2 in range(l):
            for diag_idx in range(l):
                A_indices.append([i, j1*l + diag_idx + 1, j2*l + diag_idx + 1])
                A_data += [1.0]
            if j1 == j2:
                b.append(1.0)
            else:
                b.append(0.0)
            b_ineq_mask.append(0.0)
            i += 1

    # constraint: objective-relevant entries of Y >= 0, written as -Y <= 0
    for j in range(C.nse):
        coord_a, coord_b = C.indices[j][0], C.indices[j][1]
        if coord_a < coord_b:
            A_indices.append([i, coord_a, coord_b])
            A_indices.append([i, coord_b, coord_a])
            A_data += [-0.5, -0.5]
            b.append(0.0)
            b_ineq_mask.append(1.0)
            i += 1
        elif coord_a == coord_b:
            A_indices.append([i, coord_a, coord_a])
            A_data += [-1.0]
            b.append(0.0)
            b_ineq_mask.append(1.0)
            i += 1

    # build final data structures
    A_indices = jnp.array(A_indices)
    A_data = jnp.array(A_data)
    b = jnp.array(b)
    b_ineq_mask = jnp.array(b_ineq_mask)

    return A_indices, A_data, b, b_ineq_mask


def create_qap_round(l: int, D: Array, W: BCOO) -> Callable[[BCOO, Array, Array], float]:
    @jax.jit
    def qap_round(C: BCOO, Omega: Array, P: Array) -> float:
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
    return qap_round


def get_hparams():
    parser = argparse.ArgumentParser() 
    parser.add_argument("--data_path", type=str, required=True, help="path to mat file")
    parser.add_argument("--max_iters", type=int, required=True,
                        help="number of iterations to run solver")
    parser.add_argument("--k_curr", type=int, default=1,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--k_past", type=int, default=0,
                        help="number of new eigenvectors to compute")
    parser.add_argument("--trace_factor", type=float, default=1.0,
                        help="how much space to give trace")
    parser.add_argument("--rho", type=float, default=0.1,
                        help="proximal parameter")
    parser.add_argument("--beta", type=float, default=0.25,
                        help="sufficient decrease parameter")
    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # get experiment hparams and print them out
    hparams = get_hparams()
    print(json.dumps(vars(hparams), indent=4))

    #DATAFILE = "data/qap/qapdata/chr12a.dat"
    #DATAFILE = "data/qap/tspdata/ulysses16.tsp"
    #DATAFILE = "data/qap/tspdata/dantzig42.tsp"
    #DATAFILE = "data/qap/tspdata/bayg29.tsp"
    #DATAFILE = "data/qap/tspdata/bays29.tsp"
    #DATAFILE = "data/qap/tspdata/att48.tsp"

    DATAFILE = hparams.data_path

    num_drop = 1

    if DATAFILE.split(".")[-1] == "dat":
        l, D, W, C = load_and_process_qap(DATAFILE, num_drop=num_drop)
    elif DATAFILE.split(".")[-1] == "tsp":
        l, D, W, C = load_and_process_tsp(DATAFILE, num_drop=num_drop)
    else:
        raise ValueError("Invalid data file type.")

    A_indices, A_data, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = jnp.zeros((m,))
    SCALE_A = SCALE_A.at[A_indices[:,0]].add(A_data**2)
    SCALE_A = 1.0 / jnp.sqrt(SCALE_A)
    #SCALE_X = 1.0
    #SCALE_C = 1.0
    #SCALE_A = jnp.ones((m,))

    scaled_C = BCOO((C.data * SCALE_C, C.indices), shape=C.shape)
    scaled_b = b * SCALE_X * SCALE_A
    scaled_A_data = A_data * SCALE_A.at[A_indices[:,0]].get()

    X = jnp.zeros((n, n))
    Omega = None
    P = None
    #X = None
    #Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, l))
    #P = jnp.zeros_like(Omega)
    y = jnp.zeros((m,))
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    trace_ub = hparams.trace_factor * float(l + 1) * SCALE_X

    k_curr = hparams.k_curr
    k_past = hparams.k_past

    qap_round = create_qap_round(l, D, W)

    X, P, y, z, primal_obj, tr_X = specbm(
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
        A_data=scaled_A_data,
        A_indices=A_indices,
        b=scaled_b,
        b_ineq_mask=b_ineq_mask,
        Omega=Omega,
        rho=hparams.rho,
        beta=hparams.beta,
        k_curr=k_curr,
        k_past=k_past,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A,
        eps=1e-5,  # hparams.eps,
        max_iters=hparams.max_iters,  # hparams.max_iters,
        lanczos_inner_iterations=min(n, 32),
        lanczos_max_restarts=100,  # hparams.lanczos_max_restarts,
        subprob_eps=1e-7,
        subprob_max_iters=15,
        callback_fn=qap_round)

    embed()
    exit()


    if DATAFILE.split(".")[-1] == "dat":
        l, D, W, C = load_and_process_qap(DATAFILE)
    elif DATAFILE.split(".")[-1] == "tsp":
        l, D, W, C = load_and_process_tsp(DATAFILE)
    else:
        raise ValueError("Invalid data file type.")

    A_indices, A_data, b, b_ineq_mask = get_all_problem_data(C)
    n = C.shape[0]
    m = b.shape[0]

    SCALE_X = 1.0 / float(l + 1)
    SCALE_C = 1.0 / jnp.linalg.norm(C.data)  # equivalent to frobenius norm
    SCALE_A = jnp.zeros((m,))
    SCALE_A = SCALE_A.at[A_indices[:,0]].add(A_data**2)
    SCALE_A = 1.0 / jnp.sqrt(SCALE_A)
    #SCALE_X = 1.0
    #SCALE_C = 1.0
    #SCALE_A = jnp.ones((m,))

    scaled_C = BCOO((C.data * SCALE_C, C.indices), shape=C.shape)
    scaled_b = b * SCALE_X * SCALE_A
    scaled_A_data = A_data * SCALE_A.at[A_indices[:,0]].get()

    #X = jnp.zeros((n, n))
    #Omega = None
    #P = None
    X = None
    Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, l))
    P = jnp.zeros_like(Omega)
    y = jnp.zeros((m,))
    z = jnp.zeros((m,))
    tr_X = 0.0
    primal_obj = 0.0

    trace_ub = hparams.trace_factor * float(l + 1) * SCALE_X

    k_curr = hparams.k_curr
    k_past = hparams.k_past

    qap_round = create_qap_round(l, D, W)

    X, P, y, z, primal_obj, tr_X = specbm(
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
        A_data=scaled_A_data,
        A_indices=A_indices,
        b=scaled_b,
        b_ineq_mask=b_ineq_mask,
        Omega=Omega,
        rho=hparams.rho,
        beta=hparams.beta,
        k_curr=k_curr,
        k_past=k_past,
        SCALE_C=SCALE_C,
        SCALE_X=SCALE_X,
        SCALE_A=SCALE_A,
        eps=1e-5,  # hparams.eps,
        max_iters=hparams.max_iters,  # hparams.max_iters,
        lanczos_inner_iterations=min(n, 32),
        lanczos_max_restarts=100,  # hparams.lanczos_max_restarts,
        subprob_eps=1e-7,
        subprob_max_iters=15,
        callback_fn=qap_round)