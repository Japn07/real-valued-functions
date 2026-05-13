"""
math_engine.py  –  Core computation module for the Real-Valued Functions Applet.

Provides:
  • safe expression parsing  (parse_function)
  • domain-aware evaluation   (evaluate_safe)
  • numerical gradient / Hessian / directional derivative
  • critical-point detection via scipy.optimize
"""

import numpy as np
import warnings
from scipy import optimize

# ── Safe namespace for eval() ────────────────────────────────────────────────
SAFE_NAMESPACE = {
    # Trigonometric
    'sin':    np.sin,
    'cos':    np.cos,
    'tan':    np.tan,
    'arcsin': np.arcsin,
    'arccos': np.arccos,
    'arctan': np.arctan,
    # Hyperbolic
    'sinh':   np.sinh,
    'cosh':   np.cosh,
    'tanh':   np.tanh,
    # Exponential / logarithmic
    'exp':    np.exp,
    'sqrt':   np.sqrt,
    'ln':     np.log,           # natural log
    'log':    np.log,           # alias  (1-arg)
    'log2':   np.log2,
    'log10':  np.log10,
    # Constants
    'pi':     np.pi,
    'e':      np.e,
    # Misc
    'abs':    np.abs,
}


# ── Parsing ──────────────────────────────────────────────────────────────────

def parse_function(expr_str: str, variables=('x', 'y')):
    """
    Parse a user-entered math expression string into a callable.

    Parameters
    ----------
    expr_str : str
        e.g.  "x**2 + sin(y)"
    variables : tuple of str
        Variable names, e.g. ('x', 'y') or ('x', 'y', 'z').

    Returns
    -------
    callable  that accepts arrays for each variable.
    """
    expr_str = expr_str.strip()
    # Compile once so repeated calls are fast
    code = compile(expr_str, '<user_function>', 'eval')

    def func(*args):
        ns = dict(SAFE_NAMESPACE)
        for name, val in zip(variables, args):
            ns[name] = val
        return eval(code, {"__builtins__": {}}, ns)

    return func


def parse_implicit_function(expr_str: str, variables=('x', 'y')):
    """
    Parse a user-entered math expression, supporting implicit equations.

    If *expr_str* contains '=', it is split at '=' and rewritten to
    ``LHS - (RHS)`` so that the zero set represents the implicit curve/surface.
    Otherwise delegates to :func:`parse_function` directly.

    Returns
    -------
    (callable, bool)
        The callable and a flag indicating whether the expression was implicit.
    """
    expr_str = expr_str.strip()
    if '=' in expr_str:
        parts = expr_str.split('=', 1)
        lhs = parts[0].strip()
        rhs = parts[1].strip()
        if not rhs or rhs == '0':
            rewritten = lhs
        elif not lhs or lhs == '0':
            rewritten = f'-({rhs})'
        else:
            rewritten = f'({lhs}) - ({rhs})'
        return parse_function(rewritten, variables), True
    return parse_function(expr_str, variables), False
def parse_domain_value(expr_str: str) -> float:
    """
    Parse a mathematical expression string into a float,
    allowing use of math constants (e.g., 'pi', 'e') and functions.
    """
    expr_str = expr_str.strip()
    if not expr_str:
        raise ValueError("Empty string")
    return float(eval(expr_str, {"__builtins__": {}}, SAFE_NAMESPACE))


# ── Domain-safe evaluation ───────────────────────────────────────────────────

def evaluate_safe(func, *grids):
    """
    Evaluate *func* on the supplied grids, replacing invalid values with NaN.

    Handles:
      • division by zero       → NaN  (discontinuity)
      • complex results        → NaN  (outside valid domain)
      • negative log arguments → NaN  (outside valid domain)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            result = func(*grids)
        except Exception:
            return np.full_like(grids[0], np.nan, dtype=float)

    result = np.asarray(result, dtype=complex)

    # Mask complex results
    mask = np.isnan(result.real) | np.isinf(result.real) | (result.imag != 0)
    out = result.real.astype(float)
    out[mask] = np.nan
    return out


# ── Numerical differentiation ────────────────────────────────────────────────

def numerical_gradient(func, x, y, h=1e-7):
    """
    Compute ∇f = (∂f/∂x, ∂f/∂y) via central finite differences.
    """
    df_dx = (evaluate_safe(func, x + h, y) - evaluate_safe(func, x - h, y)) / (2 * h)
    df_dy = (evaluate_safe(func, x, y + h) - evaluate_safe(func, x, y - h)) / (2 * h)
    return df_dx, df_dy


def numerical_gradient_scalar(func, x0, y0, h=1e-7):
    """
    Gradient at a single point (returns a 2-element array).
    """
    x0, y0 = float(x0), float(y0)
    df_dx = (evaluate_safe(func, np.array([x0 + h]), np.array([y0]))[0]
             - evaluate_safe(func, np.array([x0 - h]), np.array([y0]))[0]) / (2 * h)
    df_dy = (evaluate_safe(func, np.array([x0]), np.array([y0 + h]))[0]
             - evaluate_safe(func, np.array([x0]), np.array([y0 - h]))[0]) / (2 * h)
    return np.array([df_dx, df_dy])


def numerical_hessian(func, x0, y0, h=1e-5):
    """
    Compute the 2×2 Hessian matrix at a single point (x0, y0).
    """
    x0, y0 = float(x0), float(y0)

    def f(x, y):
        return evaluate_safe(func, np.array([x]), np.array([y]))[0]

    fxx = (f(x0 + h, y0) - 2 * f(x0, y0) + f(x0 - h, y0)) / h**2
    fyy = (f(x0, y0 + h) - 2 * f(x0, y0) + f(x0, y0 - h)) / h**2
    fxy = (f(x0 + h, y0 + h) - f(x0 + h, y0 - h)
           - f(x0 - h, y0 + h) + f(x0 - h, y0 - h)) / (4 * h**2)
    return np.array([[fxx, fxy],
                     [fxy, fyy]])


def directional_derivative(func, x0, y0, direction):
    """
    Compute D_u f(x0, y0) = ∇f · û  where *direction* is (dx, dy).
    """
    grad = numerical_gradient_scalar(func, x0, y0)
    u = np.array(direction, dtype=float)
    u_hat = u / (np.linalg.norm(u) + 1e-30)
    return float(np.dot(grad, u_hat))


# ── Critical-point detection ─────────────────────────────────────────────────

def _classify_critical_point(func, x0, y0):
    """
    Classify a critical point via the second-derivative test.
    Returns one of: 'local_min', 'local_max', 'saddle', 'inconclusive'.
    """
    H = numerical_hessian(func, x0, y0)
    eigvals = np.linalg.eigvalsh(H)
    det = eigvals[0] * eigvals[1]

    if det > 0:
        if eigvals[0] > 0:
            return 'local_min'
        else:
            return 'local_max'
    elif det < 0:
        return 'saddle'
    else:
        return 'inconclusive'


def find_critical_points(func, x_range, y_range, n_starts=12):
    """
    Attempt to locate critical points of *func* in the given ranges.

    Minimizes the squared norm of the gradient to find all critical points
    (minima, maxima, and saddle points) from a grid of initial points.

    Returns
    -------
    list of dict  with keys  'x', 'y', 'z', 'type'
    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    def grad_sq(xy):
        g = numerical_gradient_scalar(func, xy[0], xy[1])
        if not np.isfinite(g[0]) or not np.isfinite(g[1]):
            return 1e18
        return float(g[0]**2 + g[1]**2)

    found = []
    seen = set()

    xs = np.linspace(xmin, xmax, n_starts)
    ys = np.linspace(ymin, ymax, n_starts)

    for x0 in xs:
        for y0 in ys:
            try:
                res = optimize.minimize(grad_sq, [x0, y0],
                                        bounds=[(xmin, xmax), (ymin, ymax)],
                                        method='L-BFGS-B')
                # Check if it converged to a critical point (|grad|^2 is small)
                if res.success and res.fun < 1e-5:
                    px, py = round(res.x[0], 6), round(res.x[1], 6)
                    key = (px, py)
                    if key not in seen:
                        seen.add(key)
                        pz = evaluate_safe(func, np.array([px]), np.array([py]))[0]
                        if np.isfinite(pz):
                            cp_type = _classify_critical_point(func, px, py)
                            found.append({'x': px, 'y': py, 'z': pz, 'type': cp_type})
            except Exception:
                pass

    return found
