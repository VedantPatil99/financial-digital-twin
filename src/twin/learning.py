"""
The learning loop — what makes this a twin rather than a model.

A model predicts. A twin is *coupled to* the thing it mirrors: it predicts,
reality happens, the gap is measured, and the model updates. Without that
feedback there is no twinning, only simulation.

Three mechanisms
----------------
**Drift.** At each historical quarter the twin is re-fitted on data available
strictly up to that point and asked to project the next quarter. Comparing
against what was subsequently filed gives a drift series spanning the whole
sample — a real record of twin-versus-reality, not a synthetic one.

**Elasticity evolution.** The causal coefficients are re-estimated on an
expanding window, so you can watch them move as the firm changes. Apple's
revenue→COGS elasticity drifting is not noise; it is the operating leverage of
the business changing, and a twin that silently holds it fixed is quietly
wrong.

**Structural break detection.** When drift exceeds its own historical
distribution the relationship the twin encodes has stopped holding. That is a
finding, not a failure: it says the firm's structure changed, and dates when.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import build_features                            # noqa: E402
from graph.causal_graph import fit_graph                       # noqa: E402
from twin.statements import calibrate, opening_state, step     # noqa: E402
from twin.dynamics import (estimate_lags, baseline_path,       # noqa: E402
                           propagate_period)

LOG_PATH = os.path.join(ROOT, "data", "cache", "prediction_log.csv")


# ---------------------------------------------------------------------------
# Prediction log
# ---------------------------------------------------------------------------

def build_drift_log(panel: pd.DataFrame, macro: pd.DataFrame,
                    cashflow: pd.DataFrame, income_extras: pd.DataFrame,
                    min_train: int = 32,
                    metrics=("revenue", "gross_margin", "operating_margin",
                             "operating_income", "net_income")) -> pd.DataFrame:
    """
    Walk the history: fit on everything up to quarter t, project t+1, and
    compare against the filing that actually arrived.

    This is the twin's own track record. It is expensive (a full refit per
    quarter) but it is the honest way to say how far the twin has drifted from
    reality at any point in time.
    """
    rows: List[Dict] = []
    idx = panel.dropna(subset=["revenue", "gross_profit", "operating_income",
                               "assets", "equity"]).index

    for i in range(min_train, len(idx) - 1):
        t_now, t_next = idx[i], idx[i + 1]
        hist = panel[panel.index <= t_now]
        actual = panel.loc[t_next]

        try:
            feats_h = build_features(hist)
            macro_h = macro[macro.index <= t_now]
            graph = fit_graph(feats_h, macro_h)
            lags = estimate_lags(feats_h, macro_h)
            cf_h = cashflow[cashflow.index <= t_now]
            ie_h = income_extras[income_extras.index <= t_now]
            params = calibrate(hist, cf_h, ie_h)
            state = opening_state(hist, cf_h, ie_h)
            bl = baseline_path(hist, 1, cashflow=cf_h)
        except Exception:  # noqa: BLE001
            continue

        node_hist: Dict[str, List[float]] = {}
        dlog = propagate_period(graph, lags, [{}], node_hist, 0)
        drivers = {k: float(bl.iloc[0][k]) * float(np.exp(dlog.get(k, 0.0)))
                   for k in ("revenue", "cogs", "opex", "receivables",
                             "inventory", "payables") if k in bl.columns}
        try:
            proj = step(state, drivers, params, period=t_next)
        except Exception:  # noqa: BLE001
            continue

        pr = proj.to_row()
        act = {
            "revenue": float(actual["revenue"]),
            "operating_income": float(actual["operating_income"]),
            "net_income": float(actual["net_income"])
            if pd.notna(actual.get("net_income")) else np.nan,
            "gross_margin": float(actual["gross_profit"] / actual["revenue"]),
            "operating_margin": float(actual["operating_income"] / actual["revenue"]),
        }

        for m in metrics:
            a, p = act.get(m, np.nan), pr.get(m, np.nan)
            if not (np.isfinite(a) and np.isfinite(p)) or a == 0:
                continue
            rows.append({
                "made_at": t_now,
                "target_period": t_next,
                "metric": m,
                "predicted": p,
                "actual": a,
                "error": p - a,
                "abs_pct_error": abs((p - a) / a) * 100,
            })

    return pd.DataFrame(rows)


def drift_summary(log: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Rolling accuracy and bias per metric — the twin's health over time."""
    out = []
    for metric, g in log.groupby("metric"):
        g = g.sort_values("target_period").copy()
        g["rolling_mape"] = g["abs_pct_error"].rolling(window, min_periods=3).mean()
        g["rolling_bias"] = (g["error"] / g["actual"] * 100).rolling(
            window, min_periods=3).mean()
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else log


def detect_breaks(log: pd.DataFrame, metric: str = "operating_margin",
                  z_threshold: float = 2.5, window: int = 12) -> pd.DataFrame:
    """
    Flag quarters where the twin's error is extreme relative to its own past.

    A break is not "the model is bad" — it is evidence the underlying
    relationship changed at that date, which is information about the firm.
    """
    g = log[log["metric"] == metric].sort_values("target_period").copy()
    if len(g) < window + 2:
        return pd.DataFrame()
    mu = g["abs_pct_error"].rolling(window, min_periods=5).mean().shift(1)
    sd = g["abs_pct_error"].rolling(window, min_periods=5).std().shift(1)
    g["z"] = (g["abs_pct_error"] - mu) / sd.replace(0, np.nan)
    breaks = g[g["z"] > z_threshold].copy()
    return breaks[["target_period", "metric", "predicted", "actual",
                   "abs_pct_error", "z"]]


# ---------------------------------------------------------------------------
# Elasticity evolution
# ---------------------------------------------------------------------------

def elasticity_history(panel: pd.DataFrame, macro: pd.DataFrame,
                       min_train: int = 32, stride: int = 1) -> pd.DataFrame:
    """
    Re-estimate the causal graph on an expanding window and record how each
    elasticity moves. This is the twin *learning* — and it makes structural
    change visible as a trend in a coefficient rather than as an unexplained
    forecast miss.
    """
    idx = panel.dropna(subset=["revenue", "gross_profit"]).index
    rows = []
    for i in range(min_train, len(idx), stride):
        t = idx[i]
        hist = panel[panel.index <= t]
        try:
            g = fit_graph(build_features(hist), macro[macro.index <= t])
        except Exception:  # noqa: BLE001
            continue
        for e in g.edges:
            if e.kind != "behavioral":
                continue
            rows.append({
                "as_of": t,
                "edge": f"{e.parent}->{e.child}",
                "elasticity": e.coef,
                "t_stat": e.tstat,
                "n": e.n,
            })
    return pd.DataFrame(rows)


def elasticity_stability(hist_df: pd.DataFrame) -> pd.DataFrame:
    """Summarise how much each edge has moved across the expanding window."""
    if hist_df.empty:
        return hist_df
    out = []
    for edge, g in hist_df.groupby("edge"):
        g = g.sort_values("as_of")
        first, last = float(g["elasticity"].iloc[0]), float(g["elasticity"].iloc[-1])
        out.append({
            "edge": edge,
            "first": first,
            "latest": last,
            "change": last - first,
            "sd": float(g["elasticity"].std()),
            "min": float(g["elasticity"].min()),
            "max": float(g["elasticity"].max()),
            "stable": bool(g["elasticity"].std() < 0.15 * max(abs(g["elasticity"].mean()), 1e-9)),
        })
    return pd.DataFrame(out).sort_values("sd", ascending=False)


if __name__ == "__main__":
    from features import load_panel
    from ingest.cashflow import build_cashflow_panel, build_income_extras

    panel = load_panel()
    macro = pd.read_csv(os.path.join(ROOT, "data", "cache", "macro_quarterly.csv"),
                        index_col=0, parse_dates=True)
    cf = build_cashflow_panel()
    ie = build_income_extras()

    print("Building the twin's own track record (refit at every quarter)…")
    log = build_drift_log(panel, macro, cf, ie)
    log.to_csv(LOG_PATH, index=False)
    print(f"  {len(log)} prediction records -> {LOG_PATH}\n")

    print("Twin accuracy by metric (one quarter ahead, no lookahead):")
    s = (log.groupby("metric")["abs_pct_error"]
            .agg(["count", "mean", "median"]).round(2)
            .rename(columns={"mean": "MAPE_%", "median": "median_%"}))
    print(s.to_string())

    d = drift_summary(log)
    d.to_csv(os.path.join(ROOT, "data", "cache", "drift_log.csv"), index=False)

    print("\nStructural breaks (operating margin, z > 2.5):")
    b = detect_breaks(log, "operating_margin")
    if b.empty:
        print("  none detected")
    else:
        b = b.copy()
        b["target_period"] = b["target_period"].dt.date
        print(b.round(2).to_string(index=False))

    print("\nElasticity evolution (expanding window)…")
    eh = elasticity_history(panel, macro, stride=2)
    eh.to_csv(os.path.join(ROOT, "data", "cache", "elasticity_history.csv"),
              index=False)
    st = elasticity_stability(eh)
    print(st.round(3).to_string(index=False))
