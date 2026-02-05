# utils.py
import sympy as sp
import numpy as np


def build_functions_from_sympy(f_str, var_names):
    """
    Build numerical functions f, grad, hess from a string and variable names.

    Parameters
    ----------
    f_str : str
        Expression of f in Sympy/Python syntax.
    var_names : list[str]
        Variable names in the desired order, e.g. ["x", "y"].

    Returns
    -------
    f : callable
        f(x) -> float
    grad : callable
        grad(x) -> np.ndarray of shape (n,)
    hess : callable
        hess(x) -> np.ndarray of shape (n, n)
    vars_sym : tuple of Sympy symbols
    """
    # Create symbolic variables
    vars_sym = sp.symbols(" ".join(var_names))

    # Symbolic function
    f_sym = sp.sympify(f_str)

    # Gradient and Hessian
    grad_sym = [sp.diff(f_sym, v) for v in vars_sym]
    hess_sym = sp.hessian(f_sym, vars_sym)

    # Turn into numerical functions
    f_l = sp.lambdify(vars_sym, f_sym, "numpy")
    grad_l = sp.lambdify(vars_sym, grad_sym, "numpy")
    hess_l = sp.lambdify(vars_sym, hess_sym, "numpy")

    def f(x):
        return float(f_l(*x))

    def grad(x):
        g = grad_l(*x)
        return np.array(g, dtype=float).reshape(-1)

    def hess(x):
        H = hess_l(*x)
        return np.array(H, dtype=float)

    return f, grad, hess, vars_sym
