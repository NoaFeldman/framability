r"""
dephasing_split.py -- balance a splittable one-qubit term between two gate types.

When a lattice is Trotterised into two inequivalent bond gates A and B and
every qubit sits on one bond of each, a one-qubit term does not have to be
shared equally between them.  For pure dephasing at rate gamma per site, any
split

    gate A carries   alpha       * gamma
    gate B carries   (1 - alpha) * gamma          0 <= alpha <= 1

adds up to the same total generator.  Dephasing lowers the framability (rate)
of the gate that carries it, so alpha trades gate A's cost against gate B's.
balance_split finds a split at which the two gates are EVEN:

    |mu_A(alpha) - mu_B(1 - alpha)|  <=  rel_tol * max(|mu_A|, |mu_B|)
                                     or  <=  abs_tol   (both at the same floor)

Model- and measure-agnostic: the caller passes two callables

    rate_a(frac) -> mu        or  (mu, info)
    rate_b(frac) -> mu        or  (mu, info)

giving the measure of gate A (resp. B) when that gate carries the fraction
`frac` of the split term, so the same search serves any model, any splittable
one-qubit term and any framability measure (fixed frame or optimised).
Nothing about the measures is computed here.

Search
------
g(alpha) = mu_A(alpha) - mu_B(1 - alpha) is non-increasing when the term helps
the gate carrying it (`decreasing=True`, the dephasing case).

  1. evaluate alpha0 (default 1/2, the even share); stop if balanced.
  2. evaluate the endpoint the sign of g points to (alpha = 1 if gate A is the
     more expensive), then the other one, until g changes sign across alpha0.
  3. Illinois (modified regula falsi) inside that bracket, falling back to
     bisection on a degenerate secant step, until balanced or max_evals.

Every evaluation costs one call of each callable (for optimised frames, two
optimisations), so the budget is counted in evaluations.  If no split meets
the tolerance -- g keeps its sign on [0, 1] (one gate is the more expensive at
every split), a rate is not finite, or the budget runs out -- the evaluated
split with the smallest relative mismatch is returned with balanced=False.
A caller whose split term vanishes (e.g. gamma = 0) should pass max_evals=1:
the split is then void and alpha0 is returned after a single evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

SPLIT_VERSION = '1.0-illinois-balance'


@dataclass
class SplitResult:
    """Outcome of balance_split: the chosen split and the full search history."""
    alpha: float                 # fraction of the term on gate A (1 - alpha on B)
    mu_a: float                  # gate A's measure at alpha
    mu_b: float                  # gate B's measure at 1 - alpha
    mismatch: float              # |mu_a - mu_b| / max(|mu_a|, |mu_b|)
    balanced: bool               # tolerance met at the chosen split
    n_evals: int                 # splits evaluated
    alphas: np.ndarray = field(repr=False)          # every alpha tried, in order
    mus_a: np.ndarray = field(repr=False)
    mus_b: np.ndarray = field(repr=False)
    info_a: Any = field(default=None, repr=False)   # rate_a's info at alpha
    info_b: Any = field(default=None, repr=False)   # rate_b's info at 1 - alpha

    @property
    def rate(self) -> float:
        """The pair's cost at the chosen split: the more expensive gate."""
        return max(self.mu_a, self.mu_b)


def relative_mismatch(mu_a: float, mu_b: float) -> float:
    """|mu_a - mu_b| / max(|mu_a|, |mu_b|); 0 when both vanish, inf if either
    is not finite."""
    if not (np.isfinite(mu_a) and np.isfinite(mu_b)):
        return float('inf')
    scale = max(abs(mu_a), abs(mu_b))
    return 0.0 if scale == 0.0 else abs(mu_a - mu_b) / scale


def is_balanced(mu_a: float, mu_b: float, rel_tol: float = 0.1,
                abs_tol: float = 1e-6) -> bool:
    """True if the two measures agree to relative error rel_tol, or to abs_tol
    in absolute terms (two rates both sitting at the floor 0 are even)."""
    if not (np.isfinite(mu_a) and np.isfinite(mu_b)):
        return False
    diff = abs(mu_a - mu_b)
    return diff <= abs_tol or diff <= rel_tol * max(abs(mu_a), abs(mu_b))


def _unpack(ret):
    """rate callable output -> (mu, info)."""
    if isinstance(ret, tuple) and len(ret) == 2:
        return float(ret[0]), ret[1]
    return float(ret), None


def balance_split(rate_a: Callable, rate_b: Callable, *, rel_tol: float = 0.1,
                  abs_tol: float = 1e-6, alpha0: float = 0.5,
                  max_evals: int = 12, decreasing: bool = True,
                  alpha_tol: float = 1e-4, verbose: bool = False) -> SplitResult:
    """Find alpha in [0, 1] at which gate A (carrying alpha of the term) and
    gate B (carrying 1 - alpha) are even to rel_tol.  See the module docstring.

    decreasing : True when carrying more of the term lowers a gate's measure
                 (dephasing), so alpha -> 1 relieves gate A.  It only orders
                 the endpoint probes; both endpoints are tried if needed.
    alpha_tol  : bracket width at which the search stops refining.
    """
    if not 0.0 <= alpha0 <= 1.0:
        raise ValueError(f'alpha0 must lie in [0, 1], got {alpha0}')
    if max_evals < 1:
        raise ValueError(f'max_evals must be >= 1, got {max_evals}')

    hist: list[tuple] = []          # (alpha, mu_a, mu_b, info_a, info_b)

    def evaluate(alpha):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        mu_a, info_a = _unpack(rate_a(alpha))
        mu_b, info_b = _unpack(rate_b(1.0 - alpha))
        hist.append((alpha, mu_a, mu_b, info_a, info_b))
        if verbose:
            print(f'  split alpha={alpha:.5f}: mu_a={mu_a:+.6e}  '
                  f'mu_b={mu_b:+.6e}  mismatch='
                  f'{relative_mismatch(mu_a, mu_b):.3e}', flush=True)
        return mu_a - mu_b, is_balanced(mu_a, mu_b, rel_tol, abs_tol)

    def seen(alpha):
        return any(abs(h[0] - alpha) <= 0.5 * alpha_tol for h in hist)

    def result():
        # Balanced splits first, then the smallest mismatch, then the earliest.
        def rank(i):
            _, ma, mb = hist[i][:3]
            return (not is_balanced(ma, mb, rel_tol, abs_tol),
                    relative_mismatch(ma, mb), i)
        alpha, ma, mb, ia, ib = hist[min(range(len(hist)), key=rank)]
        return SplitResult(alpha=alpha, mu_a=ma, mu_b=mb,
                           mismatch=relative_mismatch(ma, mb),
                           balanced=is_balanced(ma, mb, rel_tol, abs_tol),
                           n_evals=len(hist),
                           alphas=np.array([h[0] for h in hist]),
                           mus_a=np.array([h[1] for h in hist]),
                           mus_b=np.array([h[2] for h in hist]),
                           info_a=ia, info_b=ib)

    # ---- 1. the even share ------------------------------------------------
    g0, ok = evaluate(alpha0)
    if ok or max_evals == 1 or not np.isfinite(g0):
        return result()

    # ---- 2. bracket: probe the promising endpoint first -------------------
    # With `decreasing`, g > 0 (gate A the more expensive) is cured by giving
    # gate A more of the term, i.e. by moving alpha toward 1.
    toward_one = (g0 > 0) == decreasing
    bracket = None
    for end in ((1.0, 0.0) if toward_one else (0.0, 1.0)):
        if len(hist) >= max_evals:
            break
        if seen(end):
            continue
        g_end, ok = evaluate(end)
        if ok:
            return result()
        if np.isfinite(g_end) and g0 * g_end < 0:
            bracket = (alpha0, g0, end, g_end)
            break
    if bracket is None:
        return result()

    # ---- 3. Illinois inside the bracket -----------------------------------
    a, fa, b, fb = bracket
    side = 0
    while len(hist) < max_evals and abs(b - a) > alpha_tol:
        c = (a * fb - b * fa) / (fb - fa)
        if seen(c) or not (min(a, b) < c < max(a, b)):
            c = 0.5 * (a + b)            # degenerate secant step: bisect
            if seen(c):
                break
        fc, ok = evaluate(c)
        if ok or not np.isfinite(fc):
            break
        if fc * fb > 0:                  # c replaces b
            b, fb = c, fc
            if side == -1:
                fa *= 0.5
            side = -1
        elif fc * fa > 0:                # c replaces a
            a, fa = c, fc
            if side == +1:
                fb *= 0.5
            side = +1
        else:                            # fc == 0 exactly
            break
    return result()
