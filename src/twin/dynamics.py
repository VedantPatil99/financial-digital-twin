"""
Temporal dynamics: shocks that unfold over time rather than all at once.

Why lags matter
---------------
A single-period shock model says a demand collapse hits revenue, inventory and
cash simultaneously. Reality is staggered: revenue falls this quarter,
production and inventory adjust over the next one or two, payables unwind after
that, and the cash trough arrives several quarters after the shock itself. The
lags *are* the risk — a firm with enough cash to absorb an instantaneous hit can
still fail if the trough is deep and late.

Two mechanisms
--------------
1. **Edge lags.** Each behavioural edge gets an integer lag in quarters,
   estimated by cross-correlation of the parent and child series and capped at
   `max_lag`. `revenue → receivables` typically lands at 0 (billing is
   immediate); `cogs → inventory` at 1 or more (production responds later).

2. **Shock persistence.** A shock decays geometrically at `persistence` per
   quarter, so a scenario is a *path*, not a step. persistence=0.8 means 80% of
   the shock remains after one quarter, 64% after two.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from graph.causal_graph import BEHAVIORAL, EXOGENOUS, FittedGraph  # noqa: E402
from twin.statements import (TwinState, PolicyParams, step,        # noqa: E402
                             assert_balanced)


# ---------------------------------------------------------------------------
# Lag estimation
# ---------------------------------------------------------------------------

def estimate_lags(feats: pd.DataFrame, macro: pd.DataFrame,
                  max_lag: int = 3) -> Dict[Tuple[str, str], int]:
    """
    For each behavioural edge, pick the lag maximising |correlation| between
    parent and child, over 0..max_lag quarters.

    Uses the same YoY log-difference basis the elasticities were fitted on, so
    lag and elasticity describe the same relationship.
    """
    def series(name):
        if name in ("gdp_yoy", "inflation_yoy", "rate_change"):
            return macro[name] if name in macro.columns else None
        col = f"dlog4_{name}"
        return feats[col] if col in feats.columns else None

    lags: Dict[Tuple[str, str], int] = {}
    for child, parents in BEHAVIORAL.items():
        y = series(child)
        if y is None:
            continue
        for p in parents:
            x = series(p)
            if x is None:
                lags[(p, child)] = 0        # injected shock: acts immediately
                continue
            best_lag, best_abs = 0, -np.inf
            for L in range(0, max_lag + 1):
                j = pd.concat([x.shift(L).rename("x"), y.rename("y")],
                              axis=1).replace([np.inf, -np.inf], np.nan).dropna()
                if len(j) < 16:
                    continue
                c = j["x"].corr(j["y"])
                if np.isfinite(c) and abs(c) > best_abs:
                    best_abs, best_lag = abs(c), L
            lags[(p, child)] = best_lag
    return lags


# ---------------------------------------------------------------------------
# Shock paths
# ---------------------------------------------------------------------------

@dataclass
class ShockPath:
    """An exogenous shock as a path over the horizon, not a single value."""
    magnitude: float
    persistence: float = 0.8      # fraction remaining each quarter
    onset: int = 0                # quarters until it starts
    ramp: int = 0                 # quarters to reach full magnitude

    def value_at(self, t: int) -> float:
        if t < self.onset:
            return 0.0
        u = t - self.onset
        if self.ramp > 0 and u < self.ramp:
            scale = (u + 1) / (self.ramp + 1)
        else:
            scale = 1.0
        decay = self.persistence ** max(u - self.ramp, 0)
        return self.magnitude * scale * decay

    def path(self, horizon: int) -> np.ndarray:
        return np.array([self.value_at(t) for t in range(horizon)])


# ---------------------------------------------------------------------------
# Baseline path
# ---------------------------------------------------------------------------

def baseline_path(panel: pd.DataFrame, horizon: int,
                  drift_window: int = 4, damping: float = 0.7,
                  cashflow: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    The counterfactual "no shock" path the scenario is measured against.

    A flat projection would be wrong twice over: it removes trend growth (so a
    shock looks cheaper than it is, because the foregone growth is invisible)
    and it removes seasonality (so a projected fiscal Q1 sits ~50% below where
    the holiday quarter actually lands).

    Both are fixed by reusing the model that won the walk-forward backtest:
    seasonal-momentum, i.e. the same quarter last year carried forward by
    recent damped YoY growth. The baseline is therefore not an assumption —
    it is the project's own best-validated forecaster (7.6% MAPE on revenue),
    which keeps the simulation and forecasting halves consistent.
    """
    rev = panel["revenue"].dropna()
    ly = np.log(rev)
    drift = float((ly.diff(4).tail(drift_window)).mean()) * damping
    if not np.isfinite(drift):
        drift = 0.0

    # cost structure: recent medians, so margins start where they actually are
    rec = panel.dropna(subset=["revenue", "cogs", "gross_profit",
                               "operating_income"]).tail(8)
    cogs_ratio = float((rec["cogs"] / rec["revenue"]).median())
    # project CASH opex: step() adds depreciation from the PP&E balance, so
    # including D&A here would double-count it.
    opex_last = float((rec["gross_profit"] - rec["operating_income"]).iloc[-1])
    if cashflow is not None and "dna" in cashflow.columns:
        d = cashflow["dna"].dropna()
        if len(d):
            common = rec.index.intersection(d.index)
            if len(common):
                opex_last -= float(d.loc[common].iloc[-1])

    wc = panel.dropna(subset=["receivables", "inventory", "payables",
                              "revenue", "cogs"]).tail(8)
    ar_ratio = float((wc["receivables"] / wc["revenue"]).median())
    inv_ratio = float((wc["inventory"] / wc["cogs"]).median())
    ap_ratio = float((wc["payables"] / wc["cogs"]).median())

    rows = []
    hist = list(rev.values)
    for t in range(horizon):
        # same quarter last year (4 back in the extending series) times drift
        prev = hist[-4] if len(hist) >= 4 else hist[-1]
        r = prev * float(np.exp(drift))
        hist.append(r)
        c = r * cogs_ratio
        rows.append({
            "revenue": r,
            "cogs": c,
            "opex": opex_last * float(np.exp(drift * (t + 1) / 4.0)),
            "receivables": r * ar_ratio,
            "inventory": c * inv_ratio,
            "payables": c * ap_ratio,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trajectory simulation
# ---------------------------------------------------------------------------

def propagate_period(graph: FittedGraph,
                     lags: Dict[Tuple[str, str], int],
                     shock_hist: List[Dict[str, float]],
                     node_hist: Dict[str, List[float]],
                     t: int) -> Dict[str, float]:
    """
    Log-deviation of every node at time t, reading each parent at its own lag.

    The subtle part is the zero-lag case. `node_hist` only holds *completed*
    periods, so a zero-lag parent's value for period t is not in it yet — it is
    being computed in this very pass. Nodes are therefore visited in
    topological order and a zero-lag parent is read from the in-progress `dlog`
    dict, falling back to history only for genuine lags.

    Getting this wrong is silent: the lookup simply misses, the parent
    contributes 0, and the shock never reaches its children. The scenario still
    runs and still balances — it just quietly does nothing. This function is
    shared by the forward simulator and the replay validator so the two cannot
    drift apart.
    """
    from graph.causal_graph import topological_order

    dlog: Dict[str, float] = {}
    for node in topological_order():
        if node in EXOGENOUS:
            dlog[node] = (float(np.log1p(shock_hist[t].get(node, 0.0)))
                          if t < len(shock_hist) else 0.0)
            continue
        total = 0.0
        for e in graph.parents_of(node):
            L = lags.get((e.parent, node), 0)
            src = t - L
            if src < 0:
                continue
            if e.parent in EXOGENOUS:
                pv = (float(np.log1p(shock_hist[src].get(e.parent, 0.0)))
                      if src < len(shock_hist) else 0.0)
            else:
                h = node_hist.get(e.parent, [])
                if src < len(h):
                    pv = h[src]
                elif L == 0:
                    pv = dlog.get(e.parent, 0.0)   # same period, already computed
                else:
                    pv = 0.0
            total += e.coef * pv
        dlog[node] = total
    return dlog


def simulate_trajectory(graph: FittedGraph,
                        state0: TwinState,
                        params: PolicyParams,
                        shocks: Dict[str, ShockPath],
                        lags: Optional[Dict[Tuple[str, str], int]] = None,
                        horizon: int = 8,
                        check_balance: bool = True,
                        baseline: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Roll the twin forward `horizon` quarters under a shock path.

    `baseline` is the no-shock path (see `baseline_path`); the shock is applied
    multiplicatively on top of it. Passing None holds the last actual quarter
    flat, which is only appropriate for isolating a shock's pure effect.

    Returns one row per projected quarter with the full three-statement output
    and derived ratios.
    """
    lags = lags or {}
    shock_hist = [{k: sp.value_at(t) for k, sp in shocks.items()}
                  for t in range(horizon)]

    # node log-deviation history, so lagged parents can be read back
    hist: Dict[str, List[float]] = {n: [] for n in BEHAVIORAL}

    from graph.causal_graph import topological_order
    order = topological_order()

    base = {
        "revenue": state0.revenue, "cogs": state0.cogs, "opex": state0.opex,
        "receivables": state0.receivables, "inventory": state0.inventory,
        "payables": state0.payables,
    }

    state = state0
    rows: List[Dict[str, float]] = []

    for t in range(horizon):
        dlog = propagate_period(graph, lags, shock_hist, hist, t)

        for node in BEHAVIORAL:
            hist.setdefault(node, []).append(dlog.get(node, 0.0))

        # shock is a multiplicative deviation from the baseline path
        if baseline is not None and t < len(baseline):
            level = {k: float(baseline.iloc[t][k]) for k in base
                     if k in baseline.columns}
        else:
            level = base
        drivers = {k: level.get(k, base[k]) * float(np.exp(dlog.get(k, 0.0)))
                   for k in base if np.isfinite(base[k])}

        period = state0.period + pd.DateOffset(days=91 * (t + 1))
        state = step(state, drivers, params, period=period)
        if check_balance:
            assert_balanced(state, tol=1e-3)

        row = state.to_row()
        row["t"] = t + 1
        for k, sp in shocks.items():
            row[f"shock_{k}"] = sp.value_at(t)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("period")
    return df


def cash_runway(traj: pd.DataFrame, floor: float = 0.0) -> Optional[int]:
    """First projected quarter (1-indexed) where cash falls below `floor`."""
    below = traj.index[traj["cash"] < floor]
    if len(below) == 0:
        return None
    return int(traj.loc[below[0], "t"])


def trough(traj: pd.DataFrame, col: str = "operating_income") -> Dict[str, float]:
    """Depth and timing of the worst point in a trajectory."""
    if col not in traj.columns or traj[col].isna().all():
        return {}
    i = traj[col].idxmin()
    return {"trough_value": float(traj.loc[i, col]),
            "trough_quarter": int(traj.loc[i, "t"]),
            "trough_period": i}
