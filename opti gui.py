# opti_gui.py
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from utils import build_functions_from_sympy
from newton_ls import newton_with_line_search
from quasi_newton_dfp import quasi_newton_dfp
from bfgs import bfgs


METHOD_DESCRIPTIONS = {
    "Newton with line search": (
        "• Uses the gradient and the Hessian of f.\n"
        "• Direction d_k is obtained by solving H(x_k) d_k = -∇f(x_k).\n"
        "• A line search chooses a step α so that f(x_k + α d_k) decreases.\n"
        "• Very fast near a minimum, but can be sensitive to the starting point."
    ),
    "Quasi-Newton (DFP) with line search": (
        "• Avoids computing the exact Hessian at each step.\n"
        "• Keeps an approximation D_k ≈ H(x_k)^{-1}, updated using gradient differences.\n"
        "• Direction: d_k = -D_k ∇f(x_k), step α from line search.\n"
        "• Cheaper per iteration than Newton; more sensitive to initialization."
    ),
    "Quasi-Newton (BFGS) with line search": (
        "• Also approximates the inverse Hessian (BFGS update).\n"
        "• Direction: d_k = -D_k ∇f(x_k), step α from line search.\n"
        "• In practice, often the most robust and efficient Quasi-Newton method."
    ),
}

EXAMPLES = {
    "Custom": "",
    "PW02 (a)": "x**2 - 5*x*y + y**4 - 25*x - 8*y",
    "PW02 (b)": "(x**4 - 3) + y**4",
    "PW02 (c)": "x**4 - 4*y**3 + 6*(x**2 + y**2) - 4*(x + y)",
}


class OptiApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Unconstrained Optimization – GALIZRA Fethi Imadeddine")
        self.geometry("1150x700")
        self.configure(bg="#f5f5f5")

        # Theme
        try:
            style = ttk.Style(self)
            if "clam" in style.theme_names():
                style.theme_use("clam")
        except Exception:
            pass

        self.create_widgets()

    def create_widgets(self):
        # ---------- Header ----------
        header = tk.Frame(self, bg="#1f4e79", pady=10)
        header.pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            header,
            text="Unconstrained Optimization Explorer",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#1f4e79",
        ).pack()

        tk.Label(
            header,
            text="Newton and Quasi-Newton Methods with Line Search",
            font=("Segoe UI", 11),
            fg="white",
            bg="#1f4e79",
        ).pack()

        tk.Label(
            header,
            text="Student: GALIZRA Fethi Imadeddine  |  Group: G3 – L3 Mathematics  |  Supervisor: Dr. N. Meddah",
            font=("Segoe UI", 9),
            fg="white",
            bg="#1f4e79",
        ).pack(pady=(4, 0))

        # ---------- Main body ----------
        body = tk.Frame(self, bg="#f5f5f5", padx=10, pady=10)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg="#f5f5f5")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))

        right = tk.Frame(body, bg="#f5f5f5")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # ===== Left: Input panels =====

        # 1. Problem settings
        frm_top = ttk.LabelFrame(left, text="1. Problem settings", padding=10)
        frm_top.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(frm_top, text="Dimension n:").grid(row=0, column=0, sticky="w")
        self.n_var = tk.StringVar(value="2")
        ttk.Entry(frm_top, width=6, textvariable=self.n_var).grid(row=0, column=1, sticky="w")

        ttk.Label(frm_top, text="Variables:").grid(row=0, column=2, sticky="w", padx=(10, 0))
        self.vars_var = tk.StringVar(value="x y")
        ttk.Entry(frm_top, width=20, textvariable=self.vars_var).grid(row=0, column=3, sticky="w")

        ttk.Label(frm_top, text="Initial x0:").grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.x0_var = tk.StringVar(value="0 0")
        ttk.Entry(frm_top, width=20, textvariable=self.x0_var).grid(row=1, column=1, sticky="w", pady=(5, 0))

        ttk.Label(frm_top, text="Tolerance:").grid(row=1, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        self.tol_var = tk.StringVar(value="1e-6")
        ttk.Entry(frm_top, width=10, textvariable=self.tol_var).grid(row=1, column=3, sticky="w", pady=(5, 0))

        ttk.Label(frm_top, text="Max iterations:").grid(row=2, column=0, sticky="w", pady=(5, 0))
        self.max_iter_var = tk.StringVar(value="200")
        ttk.Entry(frm_top, width=10, textvariable=self.max_iter_var).grid(row=2, column=1, sticky="w", pady=(5, 0))

        ttk.Label(frm_top, text="Method:").grid(row=2, column=2, sticky="w", padx=(10, 0), pady=(5, 0))
        self.method_var = tk.StringVar()
        method_combo = ttk.Combobox(
            frm_top,
            textvariable=self.method_var,
            state="readonly",
            width=32,
            values=list(METHOD_DESCRIPTIONS.keys()),
        )
        method_combo.grid(row=2, column=3, sticky="w", pady=(5, 0))
        method_combo.current(0)

        # 2. Function input
        frm_func = ttk.LabelFrame(left, text="2. Objective function f(x1,...,xn)", padding=10)
        frm_func.pack(fill=tk.BOTH, expand=False)

        # Example selector
        ex_frame = tk.Frame(frm_func, bg="#f5f5f5")
        ex_frame.pack(fill=tk.X)
        ttk.Label(ex_frame, text="Load example:").pack(side=tk.LEFT)
        self.example_var = tk.StringVar(value="Custom")
        ex_combo = ttk.Combobox(
            ex_frame,
            textvariable=self.example_var,
            state="readonly",
            width=18,
            values=list(EXAMPLES.keys()),
        )
        ex_combo.pack(side=tk.LEFT, padx=(5, 0))

        def on_example_change(event):
            key = self.example_var.get()
            expr = EXAMPLES.get(key, "")
            if expr:
                self.func_text.delete("1.0", tk.END)
                self.func_text.insert("1.0", expr)

        ex_combo.bind("<<ComboboxSelected>>", on_example_change)

        examples = (
            "Examples:\n"
            "  x**2 - 5*x*y + y**4 - 25*x - 8*y\n"
            "  (x1**4 - 3) + x2**4\n"
            "Use exact variable names given above (x, y, x1, x2, ...)."
        )
        ttk.Label(frm_func, text=examples, foreground="gray").pack(anchor="w", pady=(5, 2))

        self.func_text = tk.Text(frm_func, height=4)
        self.func_text.pack(fill=tk.X)
        self.func_text.insert("1.0", "x**2 - 5*x*y + y**4 - 25*x - 8*y")

        # 3. Method explanation
        frm_desc = ttk.LabelFrame(left, text="3. Method explanation (summary)", padding=10)
        frm_desc.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.desc_label = tk.Label(
            frm_desc,
            text=METHOD_DESCRIPTIONS[self.method_var.get()],
            justify="left",
            anchor="nw",
            wraplength=360,
        )
        self.desc_label.pack(fill=tk.BOTH, expand=True)

        def on_method_change(event):
            self.desc_label.config(text=METHOD_DESCRIPTIONS[self.method_var.get()])
        method_combo.bind("<<ComboboxSelected>>", on_method_change)

        # Buttons
        btn_frame = tk.Frame(left, bg="#f5f5f5", pady=10)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Run optimization", command=self.run_optimization).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Clear results", command=self.clear_results).pack(side=tk.LEFT, padx=5)

        # ===== Right: Results =====

        # 4. Summary (fields instead of raw text)
        frm_sum = ttk.LabelFrame(right, text="4. Summary of results", padding=10)
        frm_sum.pack(fill=tk.X)

        # Left part: scalar info
        sum_left = tk.Frame(frm_sum)
        sum_left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))

        self.lbl_method = ttk.Label(sum_left, text="Method: -")
        self.lbl_method.pack(anchor="w")
        self.lbl_status = ttk.Label(sum_left, text="Status: -")
        self.lbl_status.pack(anchor="w", pady=(2, 0))
        self.lbl_reason = ttk.Label(sum_left, text="Reason: -", wraplength=350)
        self.lbl_reason.pack(anchor="w", pady=(2, 0))
        self.lbl_iters = ttk.Label(sum_left, text="Iterations: -")
        self.lbl_iters.pack(anchor="w", pady=(2, 0))
        self.lbl_fx = ttk.Label(sum_left, text="f(x*): -")
        self.lbl_fx.pack(anchor="w", pady=(2, 0))
        self.lbl_gnorm = ttk.Label(sum_left, text="||grad f(x*)||: -")
        self.lbl_gnorm.pack(anchor="w", pady=(2, 0))

        # Right part: coordinates of x*
        sum_right = tk.Frame(frm_sum)
        sum_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(sum_right, text="Approximate minimizer x*:", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self.coords_tree = ttk.Treeview(
            sum_right,
            columns=("var", "value"),
            show="headings",
            height=5,
        )
        self.coords_tree.heading("var", text="Variable")
        self.coords_tree.heading("value", text="Value")
        self.coords_tree.column("var", width=80, anchor="center")
        self.coords_tree.column("value", width=120, anchor="center")
        self.coords_tree.pack(fill=tk.X, pady=(4, 0))

        # 5. History
        frm_hist = ttk.LabelFrame(right, text="5. Iteration history (first 20)", padding=10)
        frm_hist.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        self.history_text = tk.Text(frm_hist, height=18)
        self.history_text.pack(fill=tk.BOTH, expand=True)

    # ---------- Helpers ----------

    def clear_results(self):
        self.lbl_method.config(text="Method: -")
        self.lbl_status.config(text="Status: -")
        self.lbl_reason.config(text="Reason: -")
        self.lbl_iters.config(text="Iterations: -")
        self.lbl_fx.config(text="f(x*): -")
        self.lbl_gnorm.config(text="||grad f(x*)||: -")
        for row in self.coords_tree.get_children():
            self.coords_tree.delete(row)
        self.history_text.delete("1.0", tk.END)

    # ---------- Main optimization ----------

    def run_optimization(self):
        # Read and validate inputs
        try:
            n = int(self.n_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Dimension n must be an integer.")
            return

        var_names = self.vars_var.get().split()
        if len(var_names) != n:
            messagebox.showerror(
                "Input error",
                "Number of variable names ({}) must equal n = {}.".format(len(var_names), n),
            )
            return

        f_str = self.func_text.get("1.0", tk.END).strip()
        if not f_str:
            messagebox.showerror("Input error", "Function f must not be empty.")
            return

        try:
            x0_vals = [float(v) for v in self.x0_var.get().split()]
        except ValueError:
            messagebox.showerror("Input error", "Initial point x0 must contain valid numbers.")
            return
        if len(x0_vals) != n:
            messagebox.showerror("Input error", "x0 must have {} components.".format(n))
            return
        x0 = np.array(x0_vals, dtype=float)

        try:
            tol = float(self.tol_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Tolerance must be a valid number.")
            return

        try:
            max_iter = int(self.max_iter_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Max iterations must be an integer.")
            return

        method = self.method_var.get()

        # Build functions
        try:
            f, grad, hess, vars_sym = build_functions_from_sympy(f_str, var_names)
        except Exception as e:
            messagebox.showerror("Error in function definition", str(e))
            return

        # Run method
        try:
            if method == "Newton with line search":
                x_star, it, hist, conv, reason = newton_with_line_search(
                    f, grad, hess, x0, tol=tol, max_iter=max_iter
                )
            elif method == "Quasi-Newton (DFP) with line search":
                x_star, it, hist, conv, reason = quasi_newton_dfp(
                    f, grad, x0, tol=tol, max_iter=max_iter
                )
            else:  # BFGS
                x_star, it, hist, conv, reason = bfgs(
                    f, grad, x0, tol=tol, max_iter=max_iter
                )
        except Exception as e:
            messagebox.showerror("Runtime error", str(e))
            return

        g_norm = float(np.linalg.norm(grad(x_star)))
        fx_star = f(x_star)

        # ----- Summary labels -----
        self.lbl_method.config(text=f"Method: {method}")
        self.lbl_status.config(text=f"Status: {'Converged' if conv else 'Not converged'}")
        self.lbl_reason.config(text=f"Reason: {reason}")
        self.lbl_iters.config(text=f"Iterations: {it}")
        self.lbl_fx.config(text=f"f(x*): {fx_star}")
        self.lbl_gnorm.config(text=f"||grad f(x*)||: {g_norm:.3e}")

        # ----- x* coordinates table -----
        for row in self.coords_tree.get_children():
            self.coords_tree.delete(row)
        for name, val in zip(vars_sym, x_star):
            self.coords_tree.insert("", tk.END, values=(str(name), f"{val:.6f}"))

        # ----- History -----
        self.history_text.delete("1.0", tk.END)
        if hist is not None and len(hist) > 0:
            max_show = min(20, len(hist))
            header = "k\t" + "\t".join(var_names) + "\tf(x_k)\n"
            self.history_text.insert(tk.END, header)
            self.history_text.insert(tk.END, "-" * 60 + "\n")
            for k in range(max_show):
                xk = hist[k]
                fxk = f(xk)
                coords = "\t".join("{:.4f}".format(v) for v in xk)
                line = "{}\t{}\t{:.6f}\n".format(k, coords, fxk)
                self.history_text.insert(tk.END, line)
        else:
            self.history_text.insert(tk.END, "No history available.")


if __name__ == "__main__":
    app = OptiApp()
    app.mainloop()
