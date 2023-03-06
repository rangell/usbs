import numba as nb
import numpy as np
import scipy
from scipy.linalg import eigh_tridiagonal
from typing import Callable, Tuple

from IPython import embed


def approx_min_eigen(
    M: Callable[[np.ndarray], np.ndarray],
    n: int,
    num_iters: int,
    eps: float
) -> Tuple[float, np.ndarray]:

    V = np.empty((num_iters, n))
    omegas = np.empty((num_iters,))
    rhos = np.empty((num_iters - 1,))

    v_0 = np.random.normal(size=(n,))
    V[0] = v_0 / np.linalg.norm(v_0)

    for i in range(num_iters):
        transformed_v = M(V[i])
        omegas[i] = np.dot(V[i], transformed_v)
        if i == num_iters - 1:
            break  # we have all we need
        V[i + 1] = transformed_v - (omegas[i] * V[i])
        if i > 0:
            V[i + 1] -= rhos[i - 1] * V[i - 1]
        rhos[i] = np.linalg.norm(V[i + 1])
        if rhos[i] < eps:
            break
        V[i + 1] = V[i + 1] / rhos[i]

    min_eigen_val, u = eigh_tridiagonal(
        omegas[: i + 1], rhos[:i], select="i", select_range=(0, 0)
    )
    min_eigen_vec = (u.T @ V[: i + 1])
    # renormalize for stability
    min_eigen_vec = min_eigen_vec / np.linalg.norm(min_eigen_vec)

    #max_eigen_val, u = eigh_tridiagonal(
    #    omegas[: i + 1], rhos[:i], select="i", select_range=(i, i)
    #)
    #max_eigen_vec = (u.T @ V[: i + 1]).squeeze()
    ## renormalize for stability
    #max_eigen_vec = max_eigen_vec / np.linalg.norm(max_eigen_vec)

    return min_eigen_val.squeeze(), min_eigen_vec.T

#@nb.njit(cache=True)
def cgal(
    n: np.int,
    m: np.int,
    trace_ub: np.int,
    C_innerprod: Callable[[np.ndarray], np.double],
    C_add: Callable[[np.ndarray], np.ndarray],
    C_matvec: Callable[[np.ndarray], np.ndarray],
    A_operator: Callable[[np.ndarray], np.ndarray],
    A_operator_slim: Callable[[np.ndarray], np.ndarray],
    A_adjoint: Callable[[np.ndarray], np.ndarray],
    A_adjoint_slim: Callable[[np.ndarray, np.ndarray], np.ndarray],
    proj_K: Callable[[np.ndarray], np.ndarray],
    SCALE_C: np.double,
    SCALE_X: np.double,
    eps: np.double,
    max_iters: np.int
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.ones((n, n)) * trace_ub / n
    y = np.zeros(m)
    z = A_operator(X)
    for t in range(max_iters):
        eta = 2.0 / (t + 2.0)
        grad = C_add(A_adjoint(y) + A_adjoint(z - proj_K(z + y)))
        min_eigval, min_eigvec = scipy.sparse.linalg.eigsh(grad, k=1, which="SA")

        num_iters = int((np.ceil((t + 1) ** (1 / 4)) * np.log(n)))
        num_iters = 200
        min_eigval2, min_eigvec2 = approx_min_eigen(lambda v: grad @ v, n, num_iters, eps)

        embed()
        exit()

        X_update_dir = min_eigvec @ min_eigvec.T
        X_next = (1-eta)*X + eta*X_update_dir
        z = A_operator(X)
        y_next = y + (z - proj_K(z + y))
        obj_gap = np.trace(grad @ (X - X_update_dir)) / (SCALE_X * SCALE_C)
        infeas_gap = np.max(np.abs(y - y_next) / SCALE_X)
        print("CGAL t: ", t,
              " obj_gap: ", obj_gap,
              " infeas: ", infeas_gap,
              " obj_val: ", C_innerprod(X_next))
        if obj_gap < eps and infeas_gap < eps:
            break
        X = X_next
        y = y_next
    return X, y


@nb.njit(cache=True)
def al(
    n: np.int,
    m: np.int,
    trace_ub: np.int,
    C_innerprod: Callable[[np.ndarray], np.double],
    C_add: Callable[[np.ndarray], np.ndarray],
    C_matvec: Callable[[np.ndarray], np.ndarray],
    A_operator: Callable[[np.ndarray], np.ndarray],
    A_operator_slim: Callable[[np.ndarray], np.ndarray],
    A_adjoint: Callable[[np.ndarray], np.ndarray],
    A_adjoint_slim: Callable[[np.ndarray, np.ndarray], np.ndarray],
    proj_K: Callable[[np.ndarray], np.ndarray],
    SCALE_C: np.double,
    SCALE_X: np.double,
    eps: np.double,
    max_iters: np.int
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.ones((n, n)) * trace_ub / n
    y = np.zeros(m)
    for t in range(max_iters):
        X = pgd(X=X,
                y=y,
                trace_ub=trace_ub,
                C_innerprod=C_innerprod,
                C_add=C_add,
                C_matvec=C_matvec,
                A_operator=A_operator,
                A_operator_slim=A_operator_slim,
                A_adjoint=A_adjoint,
                A_adjoint_slim=A_adjoint_slim,
                proj_K=proj_K,
                SCALE_C=SCALE_C,
                SCALE_X=SCALE_X,
                eps=eps,
                max_iters=max_iters) 
        z = A_operator(X)
        y_next = y + (z - proj_K(z + y))
        gap = np.max(np.abs(y - y_next) / SCALE_X)
        print("AL t: ", t, " err: ", gap, " infeas: ", np.linalg.norm((z - proj_K(z + y)) / SCALE_X))
        if gap < eps:
            break
        y = y_next
    return X, y


@nb.njit(cache=True)
def pgd(
    X: np.ndarray,
    y: np.ndarray,
    trace_ub: np.int,
    C_innerprod: Callable[[np.ndarray], np.double],
    C_add: Callable[[np.ndarray], np.ndarray],
    C_matvec: Callable[[np.ndarray], np.ndarray],
    A_operator: Callable[[np.ndarray], np.ndarray],
    A_operator_slim: Callable[[np.ndarray], np.ndarray],
    A_adjoint: Callable[[np.ndarray], np.ndarray],
    A_adjoint_slim: Callable[[np.ndarray, np.ndarray], np.ndarray],
    proj_K: Callable[[np.ndarray], np.ndarray],
    SCALE_C: np.double,
    SCALE_X: np.double,
    eps: np.double,
    max_iters: np.int
) -> np.ndarray:
    # TODO: check step-size issue here?
    step_size = 1.0
    max_iters = 1e4
    z = A_operator(X)
    infeas = z - proj_K(z + y)
    objective = (C_innerprod(X) / (SCALE_X * SCALE_C)
                 + np.dot(y, infeas / SCALE_X)
                 + np.linalg.norm(infeas / SCALE_X)**2)
    for i in range(max_iters):
        grad = C_add(A_adjoint(y) + A_adjoint(z - proj_K(z + y)))
        X_update_dir = X - (step_size * grad)

        eigvals, eigvecs = np.linalg.eigh(X_update_dir)

        descend_vals = np.flip(eigvals)
        weighted_vals = (descend_vals
                         + (1.0 / np.arange(1, len(descend_vals)+1))
                            * (1 - np.cumsum(descend_vals)))
        idx = np.sum(weighted_vals > 0) - 1
        offset = weighted_vals[idx] - descend_vals[idx]
        proj_descend_vals = descend_vals + offset
        proj_descend_vals = proj_descend_vals * (proj_descend_vals > 0)
        proj_eigvals = np.flip(proj_descend_vals) * trace_ub
        proj_eigvals = np.reshape(proj_eigvals, (1, -1))

        X_new = (eigvecs * proj_eigvals) @ eigvecs.T
        z_new = A_operator(X_new)
        infeas_new = z_new - proj_K(z_new + y)
        objective_new = (C_innerprod(X_new) / (SCALE_X * SCALE_C)
                         + np.dot(y, infeas_new / SCALE_X)
                         + np.linalg.norm(infeas_new / SCALE_X)**2)
        gap = np.abs(objective - objective_new)
        gap = np.max(np.abs(X - X_new) / SCALE_X)
        print("\t PGD i: ", i, " err: ", gap, " obj: ", objective_new)
        X = X_new
        z = z_new
        infeas = infeas_new
        objective = objective_new
        if gap < eps:
            break

    return X