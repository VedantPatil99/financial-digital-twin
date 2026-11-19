"""
Decision layer: levers, sensitivity ranking, and goal-seek.

This is what separates a stress-test from a decision tool. A stress test says
"under stagflation, operating income falls 44%." A decision tool answers the
question that follows: *which lever, pulled how far, gets us back above the
line — and which factor should we be watching in the first place?*

Three capabilities:

  **Levers** — policy parameters (buybacks, capex, R&D, payout, payment terms)
  expressed as decisions rather than constants, so a scenario can combine an
  external shock with a management response.

  **Tornado** — rank every factor by the size of its effect on a chosen metric,
  so attention goes to what actually matters. Cheap to compute and the single
  most useful screen before running detailed scenarios.

  **Goal-seek** — the inverse problem. Instead of "set the lever, read the
  outcome", specify the outcome and solve for the lever. Bisection on a
  monotone response, which is the honest method here: the causal graph is
  monotone in each lever, so a root is unique when one exists.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from twin.statements import PolicyParams, TwinState          # noqa: E402
from twin.dynamics import ShockPath, simulate_trajectory     # noqa: E402


# ---------------------------------------------------------------------------
# Lever registry
# ---------------------------------------------------------------------------

@dataclass
class Lever:
    """A management decision the twin can act on."""
    key: str                  # PolicyParams attribute
    label: str
    lo: float
    hi: float
    step: float
    unit: str = ""
    help: str = ""


LEVERS: List[Lever] = [
    Lever("buyback_pct_ocf", "Buyback intensity", 0.0, 1.5, 0.05, "× OCF",
          "Share of operating cash flow returned via repurchases. Apple's "
          "historical median is ~0.77 — the reason equity keeps shrinking."),
    Lever("dividend_pct_net_income", "Dividend payout", 0.0, 0.6, 0.01, "× NI",
          "Fraction of net income paid as dividends."),
    Lever("capex_pct_revenue", "Capex intensity", 0.0, 0.12, 0.005, "× revenue",
          "Investment in property, plant and equipment."),
    Lever("effective_tax_rate", "Effective tax rate", 0.0, 0.40, 0.01, "",
          "Statutory or structural tax change."),
    Lever("interest_rate_on_debt", "Cost of debt", 0.0, 0.12, 0.0025, "p.a.",
          "Refinancing into a different rate environment."),
    Lever("yield_on_cash", "Yield on cash", 0.0, 0.10, 0.0025, "p.a.",
          "Return earned on the cash and securities balance."),
    Lever("debt_issued", "Debt issued / quarter", 0.0, 20000.0, 500.0, "$M",
          "New borrowing per quarter."),
    Lever("debt_repaid", "Debt repaid / quarter", 0.0, 20000.0, 500.0, "$M",
          "Debt retired per quarter."),
    Lever("dna_pct_revenue", "D&A intensity", 0.0, 0.10, 0.002, "× revenue",
          "Depreciation and amortisation as a share of revenue."),
]

LEVER_BY_KEY = {l.key: l for l in LEVERS}


def apply_levers(params: PolicyParams, overrides: Dict[str, float]) -> PolicyParams:
    """Return a copy of `params` with the named levers overridden."""
    clean = {k: float(v) for k, v in overrides.items()
             if k in PolicyParams.__dataclass_fields__}
    return replace(params, **clean)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

METRICS = {
    "operating_income_final": ("Operating income (final quarter)",
                               lambda t: float(t["operating_income"].iloc[-1])),
    "operating_income_total": ("Operating income (cumulative)",
                               lambda t: float(t["operating_income"].sum())),
    "operating_margin_final": ("Operating margin (final quarter)",
                               lambda t: float(t["operating_margin"].iloc[-1])),
    "operating_margin_min": ("Operating margin (worst quarter)",
                             lambda t: float(t["operating_margin"].min())),
    "net_income_total": ("Net income (cumulative)",
                         lambda t: float(t["net_income"].sum())),
    "cash_final": ("Cash (final quarter)",
                   lambda t: float(t["cash"].iloc[-1])),
    "cash_min": ("Cash (lowest point)",
                 lambda t: float(t["cash"].min())),
    "fcf_total": ("Free cash flow (cumulative)",
                  lambda t: float(t["fcf"].sum())),
    "current_ratio_min": ("Current ratio (worst quarter)",
                          lambda t: float(t["current_ratio"].min())),
    "equity_final": ("Equity (final quarter)",
                     lambda t: float(t["equity"].iloc[-1])),
}


def metric_value(traj: pd.DataFrame, metric: str) -> float:
    if metric not in METRICS:
        raise KeyError(f"Unknown metric: {metric}")
    return METRICS[metric][1](traj)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

@dataclass
class TwinRunner:
    """Bundles everything needed to evaluate one configuration."""
    graph: object
    state0: TwinState
    params: PolicyParams
    lags: Dict[Tuple[str, str], int]
    baseline: Optional[pd.DataFrame] = None
    horizon: int = 8

    def run(self, shocks: Optional[Dict[str, ShockPath]] = None,
            levers: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        p = apply_levers(self.params, levers or {})
        bl = self.baseline
        if bl is not None and len(bl) < self.horizon:
            bl = None
        return simulate_trajectory(self.graph, self.state0, p, shocks or {},
                                   self.lags, horizon=self.horizon,
                                   baseline=bl)

    def evaluate(self, metric: str,
                 shocks: Optional[Dict[str, ShockPath]] = None,
                 levers: Optional[Dict[str, float]] = None) -> float:
        return metric_value(self.run(shocks, levers), metric)


# ---------------------------------------------------------------------------
# Sensitivity (tornado)
# ---------------------------------------------------------------------------

def tornado(runner: TwinRunner, metric: str = "operating_income_total",
            factors: Optional[Dict[str, float]] = None,
            base_shocks: Optional[Dict[str, ShockPath]] = None,
            persistence: float = 0.85) -> pd.DataFrame:
    """
    Rank factors by their effect on `metric`, each moved ±its own magnitude.

    Returns a frame sorted by absolute swing — the classic tornado ordering.
    Shock factors are perturbed as shock paths; lever factors as policy
    overrides, so external risks and management choices are ranked on one
    comparable scale. That comparison is the point: it answers "is this risk
    bigger than the lever I have to offset it?"
    """
    if factors is None:
        factors = {
            "demand_shock": 0.10,
            "supply_cost_shock": 0.10,
            "inflation_yoy": 0.03,
            "fx_shock": 0.10,
            "gdp_yoy": 0.03,
            "buyback_pct_ocf": 0.25,
            "capex_pct_revenue": 0.02,
            "effective_tax_rate": 0.05,
            "interest_rate_on_debt": 0.03,
            "dividend_pct_net_income": 0.10,
        }

    base_val = runner.evaluate(metric, base_shocks, None)
    rows = []

    for name, delta in factors.items():
        is_lever = name in LEVER_BY_KEY
        vals = {}
        for sign, tag in ((+1, "high"), (-1, "low")):
            if is_lever:
                cur = getattr(runner.params, name)
                lv = LEVER_BY_KEY[name]
                new = float(np.clip(cur + sign * delta, lv.lo, lv.hi))
                vals[tag] = runner.evaluate(metric, base_shocks, {name: new})
            else:
                sh = dict(base_shocks or {})
                sh[name] = ShockPath(sign * delta, persistence=persistence)
                vals[tag] = runner.evaluate(metric, sh, None)

        lo_v, hi_v = vals["low"], vals["high"]
        rows.append({
            "factor": name,
            "kind": "lever" if is_lever else "shock",
            "delta": delta,
            "low": lo_v,
            "high": hi_v,
            "base": base_val,
            "swing": abs(hi_v - lo_v),
            "downside": min(lo_v, hi_v) - base_val,
            "upside": max(lo_v, hi_v) - base_val,
        })

    df = pd.DataFrame(rows).sort_values("swing", ascending=False)
    total = df["swing"].sum()
    df["share_of_swing"] = df["swing"] / total if total > 0 else np.nan
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Goal-seek
# ---------------------------------------------------------------------------

@dataclass
class GoalSeekResult:
    lever: str
    target_metric: str
    target_value: float
    solution: Optional[float]
    achieved: Optional[float]
    feasible: bool
    message: str
    curve: pd.DataFrame


def goal_seek(runner: TwinRunner, lever: str, metric: str, target: float,
              shocks: Optional[Dict[str, ShockPath]] = None,
              n_scan: int = 21, tol: float = 1e-4,
              max_iter: int = 40) -> GoalSeekResult:
    """
    Solve for the lever value that makes `metric` equal `target`.

    A coarse scan first, both to bracket the root and to produce the response
    curve worth showing the user; then bisection inside the bracket. If the
    target lies outside the lever's achievable range the result is reported as
    infeasible with the closest attainable value — which is itself a useful
    answer ("no amount of buyback restraint saves this").
    """
    if lever not in LEVER_BY_KEY:
        raise KeyError(f"Unknown lever: {lever}")
    lv = LEVER_BY_KEY[lever]

    grid = np.linspace(lv.lo, lv.hi, n_scan)
    vals = []
    for g in grid:
        try:
            vals.append(runner.evaluate(metric, shocks, {lever: float(g)}))
        except Exception:  # noqa: BLE001
            vals.append(np.nan)
    curve = pd.DataFrame({"lever_value": grid, "metric_value": vals})

    ok = curve.dropna()
    if ok.empty:
        return GoalSeekResult(lever, metric, target, None, None, False,
                              "Simulation failed across the lever range.", curve)

    f = lambda x: runner.evaluate(metric, shocks, {lever: float(x)}) - target  # noqa: E731

    # find a sign change in the scanned values
    sv = ok["metric_value"].to_numpy() - target
    xs = ok["lever_value"].to_numpy()
    bracket = None
    for i in range(len(sv) - 1):
        if np.isfinite(sv[i]) and np.isfinite(sv[i + 1]) and sv[i] * sv[i + 1] <= 0:
            bracket = (xs[i], xs[i + 1])
            break

    if bracket is None:
        j = int(np.argmin(np.abs(sv)))
        closest_x, closest_y = float(xs[j]), float(ok["metric_value"].iloc[j])
        direction = "above" if sv[j] > 0 else "below"
        return GoalSeekResult(
            lever, metric, target, None, closest_y, False,
            f"Target not reachable with this lever alone — the closest is "
            f"{closest_y:,.2f} at {lever}={closest_x:,.4f} "
            f"({direction} target across the whole range).",
            curve)

    a, b = bracket
    fa = f(a)
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            break
        if fa * fm <= 0:
            b = m
        else:
            a, fa = m, fm
    sol = 0.5 * (a + b)
    achieved = runner.evaluate(metric, shocks, {lever: float(sol)})

    return GoalSeekResult(
        lever, metric, target, float(sol), float(achieved), True,
        f"Set {lv.label} to {sol:,.4f} {lv.unit} to reach "
        f"{METRICS[metric][0]} = {achieved:,.2f}.",
        curve)


def optimise_levers(runner: TwinRunner, levers: List[str], metric: str,
                    maximise: bool = True,
                    shocks: Optional[Dict[str, ShockPath]] = None,
                    n_restarts: int = 3) -> Dict[str, object]:
    """
    Search several levers jointly for the best achievable `metric`.

    Uses SciPy's derivative-free Powell method: the objective is a full
    simulation, so gradients are unavailable and each evaluation is expensive.
    Multiple restarts guard against a poor starting point.
    """
    from scipy.optimize import minimize

    specs = [LEVER_BY_KEY[k] for k in levers]
    bounds = [(s.lo, s.hi) for s in specs]
    sign = -1.0 if maximise else 1.0

    def obj(x):
        ov = {k: float(v) for k, v in zip(levers, x)}
        try:
            return sign * runner.evaluate(metric, shocks, ov)
        except Exception:  # noqa: BLE001
            return 1e18

    best, best_x = np.inf, None
    rng = np.random.default_rng(0)
    starts = [np.array([getattr(runner.params, k) for k in levers], float)]
    for _ in range(max(n_restarts - 1, 0)):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))

    for x0 in starts:
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            r = minimize(obj, x0, method="Powell", bounds=bounds,
                         options={"maxiter": 60, "xtol": 1e-3, "ftol": 1e-3})
            if r.fun < best:
                best, best_x = float(r.fun), np.clip(
                    r.x, [b[0] for b in bounds], [b[1] for b in bounds])
        except Exception:  # noqa: BLE001
            continue

    if best_x is None:
        return {"success": False, "message": "Optimisation failed."}

    settings = {k: float(v) for k, v in zip(levers, best_x)}
    return {
        "success": True,
        "settings": settings,
        "achieved": sign * best,
        "baseline": runner.evaluate(metric, shocks, None),
        "metric": metric,
    }
