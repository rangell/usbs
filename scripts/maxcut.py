import cvxpy as cp
import jax
import jax.numpy as jnp
from jax import lax
from jax._src.typing import Array
from jax.experimental.sparse import BCOO
import numpy as np
from pathlib import Path
import pickle
import scipy  # type: ignore
from scipy.io import loadmat  # type: ignore
from scipy.sparse import coo_matrix, csc_matrix  # type: ignore
from typing import Any, Callable, Tuple

from solver.cgal import cgal
from solver.specbm import specbm
from solver.eigen import approx_grad_k_min_eigen

from IPython import embed


def create_C_innerprod(C: BCOO) -> Callable[[Array], float]:
    @jax.jit
    def C_innerprod(X: Array) -> float:
        return jnp.trace(C @ X)
    return C_innerprod


def create_C_add(C: BCOO) -> Callable[[Array], Array]:
    dense_C = C.todense()
    @jax.jit
    def C_add(X: Array) -> Array:
        return dense_C + X
    return C_add


def create_C_matvec(C: BCOO) -> Callable[[Array], Array]:
    @jax.jit
    def C_matvec(u: Array) -> Array:
        return C @ u
    return C_matvec


def create_A_operator() -> Callable[[Array], Array]:
    @jax.jit
    def A_operator(X: Array) -> Array:
        return jnp.diag(X)
    return A_operator


def create_A_operator_slim() -> Callable[[Array, Array], Array]:
    @jax.jit
    def A_operator_slim(u: Array) -> Array:
        return u ** 2
    return A_operator_slim


def create_A_adjoint(n: int) -> Callable[[Array], Array]:
    @jax.jit
    def A_adjoint(z: Array) -> Array:
        Y = jnp.zeros((n,n))
        Y = Y.at[jnp.diag_indices(n, ndim=2)].set(z)
        return Y
    return A_adjoint


def create_A_adjoint_slim() -> Callable[[Array, Array], Array]:
    @jax.jit
    def A_adjoint(z: Array, u: Array) -> Array:
        return z * u
    return A_adjoint


def create_proj_K(n: int, SCALE_X: float) -> Callable[[Array], Array]:
    @jax.jit
    def proj_K(z: Array) -> Array:
        return jnp.ones((n,)) * SCALE_X
    return proj_K

# TODO: put this in some utils file
def create_svec_matrix(k: int) -> BCOO:
    U = np.zeros((int(0.5*k*(k+1)), k**2))
    for a, (b, c) in enumerate(list(zip(*np.tril_indices(k)))):
        if b == c:
            U[a, b*k + c] = 1.0
        else:
            U[a, b*k + c] = 1.0 / np.sqrt(2.0)
            U[a, c*k + b] = U[a, b*k + c]
    U = coo_matrix(U)
    U = BCOO((U.data, jnp.stack((U.row, U.col)).T), shape=U.shape)
    return U


def create_Q_base(m: int, k: int, U: BCOO) -> Callable[[Array], Array]:
    @jax.jit
    def Q_base(V: Array) -> Array:
        flat_outer_prod = (V.T.reshape(1, k, m) * V.T.reshape(k, 1, m)).reshape(k**2, m)
        svec_proj = U @ flat_outer_prod
        expanded_mx = svec_proj.reshape(-1, 1, m) * svec_proj.reshape(1, -1, m)
        final_mx = jnp.sum(expanded_mx, axis=-1)
        return final_mx
    return Q_base


@jax.jit
def reconstruct(Omega: Array, P: Array, approx_eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    n = Omega.shape[0]
    rho = jnp.sqrt(n) * approx_eps * jnp.linalg.norm(P, ord=2)
    P_rho = P + rho * Omega
    B = Omega.T @ P_rho
    B = 0.5 * (B + B.T)
    L = jnp.linalg.cholesky(B)
    W, Rho, _ = jnp.linalg.svd(
        jnp.linalg.lstsq(L, P_rho.T, rcond=-1)[0].T,
        full_matrices=False,  # this compresses the output to be rank `R`
    )
    Lambda = jnp.clip(Rho ** 2 - rho, 0, np.inf)
    return W, Lambda


@jax.jit
def compute_max_cut(C: BCOO, Omega: Array, P: Array) -> int:
    W, _ = reconstruct(Omega, P)
    W_bin = 2 * (W > 0).astype(float) - 1
    return jnp.max(jnp.diag(-W_bin.T @ C @ W_bin))


def solve_scs(C: csc_matrix) -> np.ndarray[Any, Any]:
    n = C.shape[0]
    X = cp.Variable((n,n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cp.diag(X) == np.ones((n,))]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(solver=cp.SCS, verbose=True)

    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X.value)
    X_scs = X.value
    return X_scs


if __name__ == "__main__":

    # variables controlling experiment
    MAT_PATH = "./data/maxcut/Gset/G1.mat"
    WARM_START = False
    WARM_START_FRAC = 0.99
    SOLVER = "specbm"           # either "specbm" or "cgal"
    K = 5                       # number of eigenvectors to compute for specbm
    R = 100                     # size of the sketch

    # print out all of the variable for this experiment
    print("MAT_PATH: ", MAT_PATH)
    print("WARM_START_FRAC: ", WARM_START_FRAC)
    print("WARM_START: ", WARM_START)
    print("SOLVER: ", SOLVER)
    print("K: ", K)
    print("R: ", R)
    print()

    jax.config.update("jax_enable_x64", True)

    # load the problem data
    problem = loadmat(MAT_PATH)
    C = problem["Problem"][0][0][1]
    n = C.shape[0]
    C = scipy.sparse.spdiags((C @ np.ones((n,1))).T, 0, n, n) - C
    C = 0.5*(C + C.T)
    C = -0.25*C
    C = C.tocsc()

    # solve with SCS if we have not already
    scs_soln_cache = str(Path(MAT_PATH).with_suffix("")) + "_scs_soln.pkl"
    if Path(scs_soln_cache).is_file():
        with open(scs_soln_cache, "rb") as f_in:
            X_scs = pickle.load(f_in)
    else:
        X_scs = solve_scs(C)
        with open(scs_soln_cache, "wb") as f_out:
            pickle.dump(X_scs, f_out)
        
    # construct the test matrix for the sketch
    Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, R))
    Omega = None

    if Omega is None:
        X = jnp.zeros((n, n))
        P = None
    else:
        X = None
        P = jnp.zeros((n, R))
    y = jnp.zeros((n,))
    z = jnp.zeros((n,))
    tr_X = 0.0
    primal_obj = 0.0

    k_curr = K
    k_past = 0
    k = k_curr + k_past

    # for interior point methods
    U = create_svec_matrix(k)

    #### Do the warm-start
    if WARM_START:
        print("\n+++++++++++++++++++++++++++++ WARM-START ++++++++++++++++++++++++++++++++++\n")

        warm_start_n = int(WARM_START_FRAC * n)
        warm_start_C = C.tolil()[:warm_start_n, :warm_start_n].tocsr()

        warm_start_m = warm_start_n
        warm_start_trace_ub = 1.0 * float(warm_start_n)

        scaled_warm_start_C = warm_start_C.tocoo().T
        scaled_warm_start_C = BCOO(
            (scaled_warm_start_C.data,
            jnp.stack((scaled_warm_start_C.row, scaled_warm_start_C.col)).T),
            shape=scaled_warm_start_C.shape)

        warm_start_C_innerprod = create_C_innerprod(scaled_warm_start_C)
        warm_start_C_add = create_C_add(scaled_warm_start_C)
        warm_start_C_matvec = create_C_matvec(scaled_warm_start_C)
        warm_start_A_operator = create_A_operator()
        warm_start_A_operator_slim = create_A_operator_slim()
        warm_start_A_adjoint = create_A_adjoint(warm_start_n)
        warm_start_A_adjoint_slim = create_A_adjoint_slim()
        warm_start_b = jnp.ones((warm_start_n,))

        if Omega is None:
            warm_start_X = jnp.zeros((warm_start_n, warm_start_n))
            warm_start_Omega = None
            warm_start_P = None
        else:
            warm_start_X = None
            warm_start_Omega = Omega[:warm_start_n, :]
            warm_start_P = jnp.zeros((warm_start_n, R))
        warm_start_y = jnp.zeros((warm_start_m,))
        warm_start_z = jnp.zeros((warm_start_n,))

        # for quadratic subproblem solved by interior point method
        Q_base = create_Q_base(warm_start_m, k, U)

        if SOLVER == "specbm":
            (warm_start_X,
             warm_start_P,
             warm_start_y,
             warm_start_z,
             warm_start_primal_obj,
             warm_start_tr_X) = specbm(
                X=warm_start_X,
                P=warm_start_P,
                y=warm_start_y,
                z=warm_start_z,
                primal_obj=0.0,
                tr_X=0.0,
                n=warm_start_n,
                m=warm_start_m,
                trace_ub=warm_start_trace_ub,
                C=warm_start_C,
                C_innerprod=warm_start_C_innerprod,
                C_add=warm_start_C_add,
                C_matvec=warm_start_C_matvec,
                A_operator=warm_start_A_operator,
                A_operator_slim=warm_start_A_operator_slim,
                A_adjoint=warm_start_A_adjoint,
                A_adjoint_slim=warm_start_A_adjoint_slim,
                Q_base=Q_base,
                U=U,
                Omega=warm_start_Omega,
                b=warm_start_b,
                rho=0.5,
                beta=0.25,
                k_curr=k_curr,
                k_past=k_past,
                SCALE_C=1.0,
                SCALE_X=1.0,
                eps=1e-4,
                max_iters=1000,
                lanczos_num_iters=200)

        elif SOLVER == "cgal":
            # TODO: fix the output here to give back the same things as specbm
            X, y = cgal(
               n=warm_start_n,
               m=warm_start_n,
               trace_ub=warm_start_trace_ub,
               C_matvec=warm_start_C_matvec,
               A_operator_slim=warm_start_A_operator_slim,
               A_adjoint_slim=warm_start_A_adjoint_slim,
               b=warm_start_b,
               beta0=1.0,
               SCALE_C=1.0,
               SCALE_X=1.0,
               eps=1e-3,
               max_iters=10000,
               lanczos_num_iters=200)
        else:
            raise ValueError("Invalid SOLVER")

        if Omega is None:
            X = X.at[:warm_start_n, :warm_start_n].set(warm_start_X)
        else:
            P = P.at[:warm_start_n, :].set(warm_start_P)
        y = y.at[:warm_start_n].set(warm_start_y)
        z = z.at[:warm_start_n].set(warm_start_z)
        tr_X = warm_start_tr_X
        primal_obj = warm_start_primal_obj

    print("\n+++++++++++++++++++++++++++++ BEGIN ++++++++++++++++++++++++++++++++++\n")

    #SCALE_X = 1.0 / float(n)
    #SCALE_C = 1.0 / scipy.sparse.linalg.norm(C, ord="fro") 
    SCALE_X = 1.0
    SCALE_C = 1.0

    scaled_C = C * SCALE_C
    scaled_C = scaled_C.tocoo().T
    scaled_C = BCOO(
        (scaled_C.data, jnp.stack((scaled_C.row, scaled_C.col)).T), shape=scaled_C.shape)
    C = C.tocoo()
    C = BCOO(
        (C.data, jnp.stack((C.row, C.col)).T), shape=C.shape)

    trace_ub = 1.0 * float(n) * SCALE_X
    m = n

    C_innerprod = create_C_innerprod(scaled_C)
    C_add = create_C_add(scaled_C)
    C_matvec = create_C_matvec(scaled_C)
    A_operator = create_A_operator()
    A_operator_slim = create_A_operator_slim()
    A_adjoint = create_A_adjoint(n)
    A_adjoint_slim = create_A_adjoint_slim()
    b = jnp.ones((n,)) * SCALE_X

    # for quadratic subproblem solved by interior point method
    Q_base = create_Q_base(m, k, U)

    if SOLVER == "specbm":
        X, P, y, z, primal_obj, tr_X = specbm(
            X=X,
            P=P,
            y=y,
            z=z,
            primal_obj=primal_obj,
            tr_X=tr_X,
            n=n,
            m=n,
            trace_ub=trace_ub,
            C=C,
            C_innerprod=C_innerprod,
            C_add=C_add,
            C_matvec=C_matvec,
            A_operator=A_operator,
            A_operator_slim=A_operator_slim,
            A_adjoint=A_adjoint,
            A_adjoint_slim=A_adjoint_slim,
            Q_base=Q_base,
            U=U,
            Omega=Omega,
            b=b,
            rho=0.5,
            beta=0.25,
            k_curr=k_curr,
            k_past=k_past,
            SCALE_C=1.0,
            SCALE_X=1.0,
            eps=1e-3,
            max_iters=1000,
            lanczos_num_iters=200)
    elif SOLVER == "cgal":
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
            C_matvec=C_matvec,
            A_operator_slim=A_operator_slim,
            A_adjoint_slim=A_adjoint_slim,
            Omega=Omega,
            b=b,
            beta0=1.0,
            SCALE_C=SCALE_C,
            SCALE_X=SCALE_X,
            eps=1e-3,
            max_iters=1200,
            lanczos_num_iters=50)

    Omega = jax.random.normal(jax.random.PRNGKey(0), shape=(n, R))
    P = X @ Omega

    # compute max cut size
    max_cut_size = compute_max_cut(C, Omega, P)

    embed()
    exit()
