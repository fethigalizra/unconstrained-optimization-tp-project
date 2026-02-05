# main.py
import numpy as np

from utils import build_functions_from_sympy
from newton_ls import newton_with_line_search
from quasi_newton_dfp import quasi_newton_dfp
from bfgs import bfgs


def ask_problem():
    print("=== Unconstrained optimization project ===")
    print("We define a multivariable function f(x1,...,xn).")
    print()

    # Dimension
    n = int(input("Enter the dimension n: "))

    # Variable names
    print("Enter variable names separated by spaces (e.g. x y or x1 x2 ...):")
    names = input("Variables: ").split()
    if len(names) != n:
        raise ValueError("Number of variable names must equal n.")

    # Function expression
    print("\nEnter the expression of f in terms of these variables.")
    print("Examples:")
    print("  x**2 - 5*x*y + y**4 - 25*x - 8*y")
    print("  (x1**4 - 3) + x2**4")
    f_str = input("f = ")

    f, grad, hess, vars_sym = build_functions_from_sympy(f_str, names)
    return f, grad, hess, vars_sym


def ask_initial_point(vars_sym):
    print("\nInitial point x0.")
    print("Variables and order:", vars_sym)
    print("Enter the coordinates separated by spaces (no parentheses or commas).")
    x0_vals = [float(v) for v in input("x0 = ").split()]
    if len(x0_vals) != len(vars_sym):
        raise ValueError("Number of coordinates must equal number of variables.")
    return np.array(x0_vals, dtype=float)


def main():
    f, grad, hess, vars_sym = ask_problem()
    x0 = ask_initial_point(vars_sym)

    print()
    tol = float(input("Tolerance (e.g. 1e-6): "))
    max_iter = int(input("Maximum iterations (e.g. 200): "))

    print("\nChoose the method:")
    print("  1 - Newton with line search")
    print("  2 - Quasi-Newton (DFP) with line search")
    print("  3 - Quasi-Newton (BFGS) with line search")
    choice = input("Your choice = ")

    if choice == "1":
        x_star, it, _, conv, reason = newton_with_line_search(
            f, grad, hess, x0, tol=tol, max_iter=max_iter
        )
        method_name = "Newton with line search"
    elif choice == "2":
        x_star, it, _, conv, reason = quasi_newton_dfp(
            f, grad, x0, tol=tol, max_iter=max_iter
        )
        method_name = "Quasi-Newton (DFP) with line search"
    elif choice == "3":
        x_star, it, _, conv, reason = bfgs(
            f, grad, x0, tol=tol, max_iter=max_iter
        )
        method_name = "BFGS with line search"
    else:
        print("Invalid choice.")
        return

    # Final report
    print("\n===== Optimization result =====")
    print(f"Method              : {method_name}")
    print(f"Variables           : {vars_sym}")
    print(f"Initial point x0    : {x0}")
    print(f"Final point x*      : {x_star}")
    print(f"Final f(x*)         : {f(x_star)}")
    print(f"Iterations performed: {it}")
    g_final = np.linalg.norm(grad(x_star))
    print(f"‖∇f(x*)‖            : {g_final:.4e}")
    print(f"Status              : {'converged' if conv else 'NOT converged'}")
    print(f"Reason              : {reason}")


if __name__ == "__main__":
    main()
