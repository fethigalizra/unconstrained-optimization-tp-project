# quasi_newton_dfp.py
import numpy as np
from line_search import line_search_backtracking


def quasi_newton_dfp(f, grad, x0, tol=1e-6, max_iter=100):
    """
    Quasi-Newton method with DFP inverse-Hessian update and line search.

    D_{k+1} = D_k + s_k s_k^T / (y_k^T s_k) - D_k y_k y_k^T D_k / (y_k^T D_k y_k)

    Returns
    -------
    x_star, it, history, converged, reason
    """
    x = np.array(x0, dtype=float)
    n = x.size
    D = np.eye(n)
    history = [x.copy()]
    g = grad(x)
    converged = False
    reason = ""

    for k in range(max_iter):
        gnorm = np.linalg.norm(g)
        if gnorm <= tol:
            converged = True
            reason = f"gradient norm {gnorm:.2e} <= tol"
            break

        d = -D @ g

        alpha, ok = line_search_backtracking(f, grad, x, d)
        if not ok:
            d = -g
            alpha, ok = line_search_backtracking(f, grad, x, d)
            if not ok:
                reason = "line search failed"
                break

        s = alpha * d
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g

        ys = np.dot(y, s)
        Dy = D @ y
        yDy = np.dot(y, Dy)

        if ys > 1e-12 and yDy > 1e-12:
            D = D + np.outer(s, s) / ys - np.outer(Dy, Dy) / yDy

        step_norm = np.linalg.norm(x_new - x)
        history.append(x_new.copy())

        if step_norm <= tol:
            x = x_new
            g = g_new
            converged = True
            reason = f"step norm {step_norm:.2e} <= tol"
            break

        x, g = x_new, g_new

    if not converged and reason == "":
        reason = "maximum iterations reached"

    return x, k + 1, np.array(history), converged, reason
