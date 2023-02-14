# Import packages.
import cvxpy as cp
import numpy as np
import scipy

from IPython import embed

# Generate a random SDP.
n = 4
p = 3
np.random.seed(1)
C = np.random.randn(n, n)
C = C + C.T
A = []
b = []
for i in range(p):
    sub_A = np.zeros((n, n))
    sub_A[:n-1, :n-1] = np.random.randn(n-1, n-1)
    sub_A[:n-1, :n-1] = sub_A[:n-1, :n-1] + sub_A[:n-1, :n-1].T
    A.append(sub_A)
    b.append(np.random.randn())

print("C: \n", C, "\n")
print("A: \n", A, "\n")
print("b: \n", b, "\n")

# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.
X = cp.Variable((n,n), symmetric=True)
# The operator >> denotes matrix inequality.
constraints = [X >> 0]
constraints += [
    cp.trace(A[i] @ X) == b[i] for i in range(len(A))
]
constraints += [ cp.trace(X) == 1 ]
prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                  constraints)
prob.solve(solver=cp.SCS)

# Print result.
print("The optimal value is", prob.value)
print("A solution X is")
print(X.value)


### BEGIN AUGMENTED LAGRANGIAN METHODS ###

def pgd(X, y, eps, max_steps):
    step_size = 0.1
    for _ in range(max_steps):
        grad = C + A_adjoint(y) + A_adjoint(A_operator(X) - b)
        X_update_dir = X - (step_size * grad)

        eigvals, eigvecs = np.linalg.eigh(X_update_dir)

        # project eigvals onto the n-dim simplex
        descend_vals = np.flip(eigvals)
        weighted_vals = (descend_vals
                         + (1.0 / np.arange(1, len(descend_vals)+1))
                            * (1 - np.cumsum(descend_vals)))
        idx = np.sum(weighted_vals > 0) - 1
        offset = weighted_vals[idx] - descend_vals[idx]
        proj_descend_vals = descend_vals + offset
        proj_descend_vals = proj_descend_vals * (proj_descend_vals > 0)
        proj_eigvals = np.flip(proj_descend_vals)

        X_new = (eigvecs * proj_eigvals[None, :]) @ eigvecs.T

        if np.max(np.abs(X_new - X)) < eps:
            break

        X = X_new
    return X

def fw(X, y, eps, max_steps):
    for i in range(max_steps):
        eta = 2.0 / (i + 2.0)
        grad = C + A_adjoint(y) + A_adjoint(A_operator(X) - b)
        _, min_eigvec = scipy.sparse.linalg.eigsh(grad, k=1, which="SA")
        X_update_dir = min_eigvec @ min_eigvec.T
        if i > 0 and np.trace(grad @ (X - X_update_dir)) < eps:
            break
        X = (1-eta)*X + eta*X_update_dir
    return X

def fw_linesearch(X, y, eps, max_steps):
    curvature_est_shinkage = 0.99
    curvature_est_expansion = 2.0

    for i in range(max_steps):
        grad = C + A_adjoint(y) + A_adjoint(A_operator(X) - b)
        _, min_eigvec = scipy.sparse.linalg.eigsh(grad, k=1, which="SA")
        X_update_dir = (min_eigvec @ min_eigvec.T) - X
        X_update_dir_norm = np.linalg.norm(X_update_dir, ord="fro")

        # Set the initial local curvature estimate
        if i == 0:
            local_curve_est = np.linalg.norm(
                A_adjoint(A_operator(X) - b) - A_adjoint(A_operator(X + eps*X_update_dir) - b))
            local_curve_est = local_curve_est / (eps * X_update_dir_norm)

        # Shrink local curvature estimate
        local_curve_est = local_curve_est * curvature_est_shinkage

        # Check convergence
        obj_gap_lb = -1 * np.trace(grad @ X_update_dir)
        if i > 0 and obj_gap_lb < eps:
            break

        # Line search
        eta = min(obj_gap_lb / (local_curve_est * X_update_dir_norm**2), 1)
        for j in range(10):
            _a = aug_Lagrangian(X + eta*X_update_dir, y)
            _b = (aug_Lagrangian(X, y) - eta*obj_gap_lb + 
                  ((local_curve_est * eta**2 * X_update_dir_norm**2) / 2))
            if _a < _b + eps:
                break
            local_curve_est = local_curve_est * curvature_est_expansion
            eta = min(obj_gap_lb / (local_curve_est * X_update_dir_norm**2), 1)
        if j > 8:
            raise AssertionError("Limit reached")

        X = X + eta*X_update_dir

    return X

eps = 1e-4

# Define operator functions for AL methods.
A_operator = lambda M : np.stack([np.trace(A[i] @ M) for i in range(len(A))])
A_adjoint = lambda v : sum([v[i] * A[i] for i in range(len(A))])
aug_Lagrangian = lambda _X, _y : (np.trace(C @ _X)
                                  + np.dot(y, A_operator(_X) - b)
                                  + np.linalg.norm(A_operator(_X) - b)**2)
b = np.asarray(b)

# Test Augmented Lagrangian Methods.
X = np.eye(n) / n
y = np.zeros_like(b)

max_steps = int(1e6)
for t in range(max_steps):
    # primal step
    X = pgd(X, y, eps, max_steps)
    #X = fw(X, y, eps, max_steps)
    #X = fw_linesearch(X, y, eps, max_steps)

    # dual step
    y_prev = y
    y = y_prev + A_operator(X) - b

    print(np.max(np.abs(y - y_prev)))

    # check convergence
    if np.max(np.abs(y - y_prev)) < eps:
        break

    

embed()
exit()
