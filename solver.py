import numba as nb
import numpy as np
from typing import Callable, Tuple


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
    # Returns X, y (for now)
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