# line_search.py
import numpy as np


def line_search_backtracking(f, grad, x, d, alpha0=1.0, c1=1e-4, tau=0.5):
    """
    Backtracking line search with Armijo condition.

    Parameters
    ----------
    f, grad : callables
    x : np.ndarray
        Current point.
    d : np.ndarray
        Search direction.
    alpha0 : float
        Initial step length.
    c1 : float
        Armijo parameter.
    tau : float
        Step reduction factor.

    Returns
    -------
    alpha : float
    ok : bool
        True if a satisfactory step was found.
    """
    alpha = alpha0
    fx = f(x)
    g = grad(x)
    gd = np.dot(g, d)

    # If not a descent direction, signal failure
    if gd >= 0:
        return 0.0, False

    while alpha > 1e-12:
        x_new = x + alpha * d
        if f(x_new) <= fx + c1 * alpha * gd:
            return alpha, True
        alpha *= tau

    return 0.0, False
