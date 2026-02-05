# newton_ls.py
import numpy as np
from line_search import line_search_backtracking


def newton_with_line_search(f, grad, hess, x0, tol=1e-6, max_iter=100):
    """
    Newton method with backtracking line search.

    x_{k+1} = x_k + alpha_k * d_k,
    d_k = -H(x_k)^{-1} grad(x_k).

    Returns
    -------
    x_star : np.ndarray
    it : int
        Number of iterations performed.
    history : np.ndarray
        Array of iterates x_k.
    converged : bool
    reason : str
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    converged = False
    reason = ""

    for k in range(max_iter):
        g = grad(x)
        gnorm = np.linalg.norm(g)
        if gnorm <= tol:
            converged = True
            reason = f"gradient norm {gnorm:.2e} <= tol"
            break

        H = hess(x)

        # Solve H d = -g (with simple regularization if needed)
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            H_reg = H + 1e-6 * np.eye(len(x))
            d = np.linalg.solve(H_reg, -g)

        alpha, ok = line_search_backtracking(f, grad, x, d)
        if not ok:
            # Try steepest descent once
            d = -g
            alpha, ok = line_search_backtracking(f, grad, x, d)
            if not ok:
                reason = "line search failed"
                break

        x_new = x + alpha * d
        step_norm = np.linalg.norm(x_new - x)
        history.append(x_new.copy())

        if step_norm <= tol:
            x = x_new
            converged = True
            reason = f"step norm {step_norm:.2e} <= tol"
            break

        x = x_new

    if not converged and reason == "":
        reason = "maximum iterations reached"

    return x, k + 1, np.array(history), converged, reason
