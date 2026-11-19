"""
The causal financial graph: the mechanism that makes this a *twin* rather than
a forecaster.

Design
------
A forecasting model predicts one variable. A digital twin has to answer "if
inflation rises 300bp, what happens to my liquidity?" -- which requires knowing
how a change *propagates* through the firm's financial structure.

We model the firm as a DAG over financial variables with two edge types, and
the distinction between them is the point:

  STRUCTURAL edges are accounting identities. Gross profit IS revenue minus
  COGS. These hold exactly, always, with coefficient 1. They are never
  estimated -- estimating them would be fitting noise to a tautology.

  BEHAVIORAL edges are empirical relationships whose strength is unknown:
  how much COGS actually moves when revenue moves, how inventory responds to
  COGS. These are estimated from Apple's own filing history as elasticities
  (regression on log-differences, so coefficients read as % per %).

Shock propagation walks the DAG in topological order, applying behavioral
elasticities to compute log-changes, then re-imposing the structural identities
so the shocked statement still balances. A scenario that broke the balance
sheet would be worthless for decision-making.

Elasticity interpretation: a coefficient of 0.9 on revenue -> COGS means a 10%
revenue rise drives a 9% COGS rise, i.e. mild operating leverage.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Graph specification
# ---------------------------------------------------------------------------

# Exogenous drivers: shocks enter the system here.
EXOGENOUS = [
    "gdp_yoy",          # real demand
    "inflation_yoy",    # input & pricing pressure
    "rate_change",      # cost of debt
    "demand_shock",     # user-controlled: idiosyncratic demand
    "supply_cost_shock",  # user-controlled: supply-chain / input cost
    "fx_shock",         # user-controlled: currency translation
]

# Behavioral edges: child <- [parents]. Estimated from data.
BEHAVIORAL: Dict[str, List[str]] = {
    "revenue":     ["gdp_yoy", "inflation_yoy", "demand_shock", "fx_shock"],
    "cogs":        ["revenue", "inflation_yoy", "supply_cost_shock"],
    "opex":        ["revenue", "inflation_yoy"],
    "inventory":   ["cogs"],
    "receivables": ["revenue"],
    "payables":    ["cogs"],
}

# Structural edges: exact accounting identities, applied after propagation.
# Each entry is (target, callable over a dict of levels).
STRUCTURAL = {
    "gross_profit":     lambda v: v["revenue"] - v["cogs"],
    "operating_income": lambda v: v["gross_profit"] - v["opex"],
    "assets_curr":      lambda v: v["cash"] + v["receivables"] + v["inventory"] + v["other_curr_assets"],
    "working_capital":  lambda v: v["assets_curr"] - v["liab_curr"],
}

# Shock variables that are user-injected rather than observed in history.
# They have no historical variation, so their elasticities are structural
# priors rather than fitted values (documented, and adjustable in the UI).
PRIOR_ELASTICITIES = {
    ("demand_shock", "revenue"): 1.00,      # a 1% demand shock moves revenue 1%
    ("fx_shock", "revenue"): 0.60,          # ~60% of Apple revenue is non-US
    ("supply_cost_shock", "cogs"): 1.00,    # input cost passes into COGS 1:1
}


@dataclass
class Edge:
    parent: str
    child: str
    kind: str                 # 'behavioral' | 'prior'
    coef: float
    se: float = np.nan
    tstat: float = np.nan
    n: int = 0


@dataclass
class FittedGraph:
    edges: List[Edge] = field(default_factory=list)
    fit_stats: Dict[str, dict] = field(default_factory=dict)

    def parents_of(self, child: str) -> List[Edge]:
        return [e for e in self.edges if e.child == child]

    def coef(self, parent: str, child: str) -> float:
        for e in self.edges:
            if e.parent == parent and e.child == child:
                return e.coef
        return 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "parent": e.parent, "child": e.child, "kind": e.kind,
            "elasticity": e.coef, "std_err": e.se, "t_stat": e.tstat, "n": e.n,
        } for e in self.edges])


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def _ols(X: np.ndarray, y: np.ndarray):
    """OLS with an intercept. Returns (betas, std_errs, t_stats, r2)."""
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    dof = max(n - Xd.shape[1], 1)
    sigma2 = float(resid @ resid) / dof
    try:
        XtX_inv = np.linalg.pinv(Xd.T @ Xd)
        se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0))
    except np.linalg.LinAlgError:
        se = np.full(Xd.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = beta / se
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else np.nan
    # drop intercept
    return beta[1:], se[1:], t[1:], r2


def _series_for(name: str, feats: pd.DataFrame, macro: pd.DataFrame,
                horizon: str = "yoy") -> Optional[pd.Series]:
    """
    Resolve a node name to the series used in regression.

    Financial levels enter as log-differences (so coefficients read as
    elasticities: % per %). Macro drivers are already YoY rates and enter
    directly -- which is why the default horizon for the financial side is
    also YoY: mixing a QoQ dependent variable with YoY regressors mismatches
    the horizons and, for a business this seasonal, destroys the signal.
    """
    if name in ("gdp_yoy", "inflation_yoy", "rate_change"):
        return macro[name] if name in macro.columns else None
    if name in ("demand_shock", "supply_cost_shock", "fx_shock"):
        return None  # injected, not observed
    prefix = "dlog4_" if horizon == "yoy" else "dlog_"
    col = f"{prefix}{name}"
    if col in feats.columns:
        return feats[col]
    return None


def fit_graph(feats: pd.DataFrame, macro: pd.DataFrame,
              min_obs: int = 20, horizon: str = "yoy") -> FittedGraph:
    """
    Estimate every behavioral edge; attach documented priors for shock nodes.

    horizon: 'yoy' (default, seasonality-differenced) or 'qoq'.
    Fiscal-quarter fixed effects are included so any residual seasonality is
    absorbed rather than loading onto the macro coefficients.
    """
    g = FittedGraph()

    for child, parents in BEHAVIORAL.items():
        y_s = _series_for(child, feats, macro, horizon)
        if y_s is None:
            continue

        observed = [p for p in parents
                    if _series_for(p, feats, macro, horizon) is not None]
        if not observed:
            continue

        df = pd.DataFrame({p: _series_for(p, feats, macro, horizon)
                           for p in observed})
        df["__y"] = y_s

        # fiscal-quarter fixed effects (Q1 is the reference level)
        if "fiscal_q" in feats.columns:
            fq = feats["fiscal_q"].reindex(df.index)
            for q in (2, 3, 4):
                df[f"__q{q}"] = (fq == q).astype(float)

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        if len(df) < min_obs:
            g.fit_stats[child] = {"n": len(df), "r2": np.nan,
                                  "note": "insufficient observations"}
            continue

        dummies = [c for c in df.columns if c.startswith("__q")]
        regressors = observed + dummies
        X = df[regressors].to_numpy(float)
        y = df["__y"].to_numpy(float)
        beta, se, t, r2 = _ols(X, y)

        # keep only the structural parents as edges; dummies are nuisance terms
        for i, p in enumerate(observed):
            g.edges.append(Edge(p, child, "behavioral", float(beta[i]),
                                float(se[i]), float(t[i]), len(df)))
        g.fit_stats[child] = {"n": len(df), "r2": float(r2),
                              "parents": observed}

    for (p, c), v in PRIOR_ELASTICITIES.items():
        g.edges.append(Edge(p, c, "prior", float(v)))

    return g


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def topological_order() -> List[str]:
    """Kahn's algorithm over the behavioral DAG."""
    nodes, incoming = set(), {}
    for child, parents in BEHAVIORAL.items():
        nodes.add(child)
        for p in parents:
            nodes.add(p)
            incoming.setdefault(child, set()).add(p)
    for n in nodes:
        incoming.setdefault(n, set())

    ready = [n for n in nodes if not incoming[n]]
    order, incoming = [], {k: set(v) for k, v in incoming.items()}
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in list(incoming):
            if n in incoming[m]:
                incoming[m].discard(n)
                if not incoming[m] and m not in order and m not in ready:
                    ready.append(m)
    if len(order) != len(nodes):
        raise ValueError("Cycle detected in causal graph")
    return order


def propagate(graph: FittedGraph, base: Dict[str, float],
              shocks: Dict[str, float]) -> Dict[str, float]:
    """
    Apply exogenous `shocks` (as fractional changes, e.g. 0.03 = +3%) to the
    `base` financial state and return the shocked state.

    Behavioral edges act in log space; structural identities are re-imposed
    afterwards so the resulting statement still balances.

    Units: callers specify shocks the way people think about them -- a
    fractional change -- while propagation runs in logs. The conversion
    happens here, at the boundary: dlog = ln(1 + shock). Feeding 0.10 in
    directly would yield exp(0.10) - 1 = 10.52%, not the 10% the caller asked
    for, and that error compounds along every downstream edge.
    """
    dlog: Dict[str, float] = {}
    for k, v in shocks.items():
        v = float(v)
        if v <= -1.0:  # a -100% shock is not representable in log space
            v = -0.9999
        dlog[k] = float(np.log1p(v))

    for node in topological_order():
        if node in dlog or node in EXOGENOUS:
            dlog.setdefault(node, 0.0)
            continue
        total = 0.0
        for e in graph.parents_of(node):
            total += e.coef * dlog.get(e.parent, 0.0)
        dlog[node] = total

    out = dict(base)
    for node in BEHAVIORAL:
        if node in base and np.isfinite(base[node]):
            out[node] = base[node] * float(np.exp(dlog.get(node, 0.0)))

    # Re-impose accounting identities.
    out.setdefault("other_curr_assets", max(
        base.get("assets_curr", 0.0) - base.get("cash", 0.0)
        - base.get("receivables", 0.0) - base.get("inventory", 0.0), 0.0))
    for target, fn in STRUCTURAL.items():
        try:
            out[target] = float(fn(out))
        except (KeyError, TypeError):
            pass

    out["_dlog"] = dlog
    return out


if __name__ == "__main__":
    from features import load_panel, build_features  # noqa: E402
    import sys
    sys.path.insert(0, os.path.join(ROOT, "src"))

    panel = load_panel()
    feats = build_features(panel)
    macro = pd.read_csv(os.path.join(ROOT, "data", "cache", "macro_quarterly.csv"),
                        index_col=0, parse_dates=True)

    g = fit_graph(feats, macro)

    print("Estimated causal graph\n" + "=" * 72)
    df = g.to_frame()
    beh = df[df.kind == "behavioral"].copy()
    beh["sig"] = np.where(beh.t_stat.abs() > 2.0, "*", " ")
    print(beh.round(3).to_string(index=False))

    print("\nPriors for injected shock nodes (not estimable from history):")
    print(df[df.kind == "prior"][["parent", "child", "elasticity"]].to_string(index=False))

    print("\nPer-equation fit:")
    for child, st in g.fit_stats.items():
        r2 = st.get("r2")
        r2s = "n/a" if r2 is None or not np.isfinite(r2) else f"{r2:.3f}"
        print(f"  {child:<14} n={st.get('n',0):>3}  R2={r2s}")

    print("\nTopological order:", " -> ".join(topological_order()))
