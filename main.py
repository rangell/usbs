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
X_scs = X.value


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

def cgal_primal_step(X, y, eps, t):
    eta = 2.0 / (t + 2.0)
    grad = C + A_adjoint(y) + A_adjoint(A_operator(X) - b)
    _, min_eigvec = scipy.sparse.linalg.eigsh(grad, k=1, which="SA")
    X_update_dir = min_eigvec @ min_eigvec.T
    if i > 0 and np.trace(grad @ (X - X_update_dir)) < eps:
        return X
    X = (1-eta)*X + eta*X_update_dir
    return X

def spec_fw(X, y, k, eps, max_steps):
    for i in range(max_steps):
        grad = C + A_adjoint(y) + A_adjoint(A_operator(X) - b)
        _, V = scipy.sparse.linalg.eigsh(grad, k=k, which="SA")

        step_size = 0.1
        S = np.eye(k) / k
        eta = 0
        for j in range(max_steps):
            grad_S = (V.T @ C @ V
                      + V.T @ A_adjoint(y) @ V
                      + V.T @ A_adjoint(A_operator((eta*X) + (V @ S @ V.T)) - b) @ V)
            grad_eta = (np.trace(C @ X)
                        + np.dot(y, A_operator(X))
                        + eta * np.linalg.norm(A_operator(X))**2
                        + np.dot(A_operator(X), A_operator(V @ S @ V.T) - b))
            S_update_dir = S - (step_size * grad_S)
            eta_update_dir = eta - (step_size * grad_eta)

            eigvals, eigvecs = np.linalg.eigh(S_update_dir)
            eta_index = np.sum(eigvals < eta_update_dir)
            eigvals = np.insert(eigvals, eta_index, eta_update_dir)

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

            eta_new = proj_eigvals[eta_index]
            proj_eigvals = np.delete(proj_eigvals, eta_index)
            S_new = (eigvecs * proj_eigvals[None, :]) @ eigvecs.T

            #print('\t', np.max(np.abs(S_new - S)), np.abs(eta - eta_new))
            if np.max(np.abs(S_new - S)) < eps and np.abs(eta - eta_new) < eps:
                break

            S = S_new
            eta = eta_new
        
        X_new = eta * X + V @ S @ V.T
        if np.max(np.abs(X - X_new)) < eps:
            break
        #print(aug_Lagrangian(X_new, y), np.max(np.abs(X - X_new)))

        X = X_new

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
    X1 = pgd(X, y, eps, max_steps)
    X2 = fw(X, y, eps, max_steps)
    #X = cgal_primal_step(X, y, eps, t)

    k = 2
    X3 = spec_fw(X, y, k, eps, max_steps)

    X = X1

    # dual step
    y_prev = y
    y = y_prev + A_operator(X) - b

    print(np.max(np.abs(y - y_prev)), np.mean(np.abs(y - y_prev)))

    # check convergence
    if np.max(np.abs(y - y_prev)) < eps:
        break

    

embed()
exit()
