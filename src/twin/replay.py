"""
Historical replay: does the twin reproduce shocks that actually happened?

Why this is the strongest test in the project
---------------------------------------------
Walk-forward backtesting validates *forecast* accuracy — can the model guess
next quarter's revenue? That is a real test, but it is not a test of the twin.
The twin's claim is causal: that a shock entering here propagates through the
firm's structure to there. A forecaster can score well while getting every
causal channel wrong.

Replay tests the causal claim directly. Rewind the twin to before a known
shock, feed it the conditions that actually occurred, roll it forward, and
compare its projected financial statements against the ones Apple actually
filed. If the twin says margin compresses 300bp and it compressed 280bp, the
mechanism is doing real work.

No lookahead
------------
The causal graph is re-fitted using only data strictly before the replay start,
so the twin knows nothing about the episode it is being asked to reproduce.
This matters: fitting on the full sample and then "predicting" 2020 would be
circular, and would flatter every result.

Two modes
---------
**macro**       Feed only the observed macro variables (GDP, inflation) and let
                the graph derive everything, revenue included. This tests the
                full chain from economy to financial statement. It is the harder
                test, and for Apple it is expected to do poorly — the revenue
                equation has R² ≈ 0.03 because product cycles, not macro, drive
                the top line. Reported honestly rather than hidden.

**conditional** Feed the actual revenue path and test whether the twin correctly
                propagates it into cost of sales, margin, working capital, cash
                and the balance sheet. This isolates the propagation mechanism
                from the demand forecast, and is the more meaningful test of a
                digital twin: given what happened to the top line, does the rest
                of the financial picture follow?
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import build_features                              # noqa: E402
from graph.causal_graph import fit_graph                         # noqa: E402
from twin.statements import calibrate, opening_state, step       # noqa: E402
from twin.dynamics import (estimate_lags, baseline_path,        # noqa: E402
                           propagate_period)


# ---------------------------------------------------------------------------
# Episodes worth replaying
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    name: str
    start: str          # last quarter the twin is allowed to know about
    horizon: int
    description: str


EPISODES: List[Episode] = [
    Episode("COVID-19", "2019-12-28", 6,
            "Pandemic onset. Global demand collapse in 2020 H1, followed by an "
            "unusually strong consumer-electronics recovery."),
    Episode("2022 inflation shock", "2021-12-25", 6,
            "CPI peaks near 9%, aggressive Fed tightening, input and logistics "
            "costs rising sharply."),
    Episode("2018-19 China / trade tension", "2018-09-29", 5,
            "Tariff escalation and a sharp iPhone demand slowdown in Greater "
            "China."),
]


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

COMPARE = ["revenue", "gross_profit", "operating_income", "net_income",
           "cash", "equity", "inventory", "receivables", "payables"]

RATIO_COMPARE = ["gross_margin", "operating_margin", "current_ratio"]


def _actual_frame(panel: pd.DataFrame, start: pd.Timestamp,
                  horizon: int) -> pd.DataFrame:
    after = panel[panel.index > start].head(horizon).copy()
    after["gross_margin"] = after["gross_profit"] / after["revenue"]
    after["operating_margin"] = after["operating_income"] / after["revenue"]
    after["current_ratio"] = after["assets_curr"] / after["liab_curr"]
    return after


def replay_episode(panel: pd.DataFrame, macro: pd.DataFrame,
                   cashflow: pd.DataFrame, income_extras: pd.DataFrame,
                   episode: Episode, mode: str = "conditional") -> Dict:
    """
    Replay one episode and score the twin against what actually happened.

    Returns a dict with the projected frame, the actual frame, and per-metric
    errors.
    """
    start = pd.Timestamp(episode.start)

    # ---- refit everything on pre-episode data only ------------------------
    hist = panel[panel.index <= start]
    if len(hist) < 24:
        return {"episode": episode, "error": "insufficient history"}

    feats_hist = build_features(hist)
    macro_hist = macro[macro.index <= start]
    graph = fit_graph(feats_hist, macro_hist)
    lags = estimate_lags(feats_hist, macro_hist)

    cf_hist = cashflow[cashflow.index <= start]
    ie_hist = income_extras[income_extras.index <= start]
    params = calibrate(hist, cf_hist, ie_hist)

    try:
        state = opening_state(hist, cf_hist, ie_hist)
    except (IndexError, KeyError) as e:  # noqa: BLE001
        return {"episode": episode, "error": f"no opening state: {e}"}

    actual = _actual_frame(panel, start, episode.horizon)
    if actual.empty:
        return {"episode": episode, "error": "no post-episode actuals"}

    horizon = len(actual)
    baseline = baseline_path(hist, horizon, cashflow=cf_hist)

    # ---- construct the driving shocks -------------------------------------
    from twin.dynamics import ShockPath, simulate_trajectory
    from graph.causal_graph import topological_order, EXOGENOUS, BEHAVIORAL

    if mode == "macro":
        # Feed the macro path that actually occurred, as deviations from the
        # pre-episode mean (the graph is fitted on deviations, not levels).
        fut_macro = macro[(macro.index > start)].head(horizon)
        base_infl = float(macro_hist["inflation_yoy"].tail(8).mean())
        base_gdp = float(macro_hist["gdp_yoy"].tail(8).mean())

        shock_hist = []
        for i in range(horizon):
            if i < len(fut_macro):
                r = fut_macro.iloc[i]
                shock_hist.append({
                    "inflation_yoy": float(r.get("inflation_yoy", base_infl)) - base_infl,
                    "gdp_yoy": float(r.get("gdp_yoy", base_gdp)) - base_gdp,
                })
            else:
                shock_hist.append({})
    else:
        # conditional: drive revenue to its actual path, propagate the rest
        shock_hist = []
        for i in range(horizon):
            actual_rev = float(actual["revenue"].iloc[i])
            base_rev = float(baseline["revenue"].iloc[i])
            dev = actual_rev / base_rev - 1.0 if base_rev else 0.0
            shock_hist.append({"demand_shock": dev})

    # ---- roll forward -----------------------------------------------------
    order = topological_order()
    node_hist: Dict[str, List[float]] = {n: [] for n in BEHAVIORAL}
    base_levels = {
        "revenue": state.revenue, "cogs": state.cogs, "opex": state.opex,
        "receivables": state.receivables, "inventory": state.inventory,
        "payables": state.payables,
    }

    rows = []
    for t in range(horizon):
        dlog = propagate_period(graph, lags, shock_hist, node_hist, t)

        for node in BEHAVIORAL:
            node_hist.setdefault(node, []).append(dlog.get(node, 0.0))

        level = {k: float(baseline.iloc[t][k]) for k in base_levels
                 if k in baseline.columns}
        drivers = {k: level.get(k, base_levels[k]) * float(np.exp(dlog.get(k, 0.0)))
                   for k in base_levels}

        state = step(state, drivers, params, period=actual.index[t])
        rows.append(state.to_row())

    proj = pd.DataFrame(rows).set_index("period")

    # ---- score ------------------------------------------------------------
    errs = []
    for col in COMPARE:
        if col not in proj.columns or col not in actual.columns:
            continue
        a = actual[col].to_numpy(float)
        p = proj[col].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(p) & (a != 0)
        if not m.any():
            continue
        errs.append({
            "metric": col,
            "actual_final": float(a[m][-1]),
            "twin_final": float(p[m][-1]),
            "MAPE_%": float(np.mean(np.abs((p[m] - a[m]) / a[m])) * 100),
            "final_err_%": float((p[m][-1] - a[m][-1]) / a[m][-1] * 100),
        })

    for col in RATIO_COMPARE:
        if col not in proj.columns or col not in actual.columns:
            continue
        a = actual[col].to_numpy(float)
        p = proj[col].to_numpy(float)
        m = np.isfinite(a) & np.isfinite(p)
        if not m.any():
            continue
        errs.append({
            "metric": col,
            "actual_final": float(a[m][-1]),
            "twin_final": float(p[m][-1]),
            "MAPE_%": float(np.mean(np.abs(p[m] - a[m])) * 10000) / 100,  # bp -> pp
            "final_err_%": float((p[m][-1] - a[m][-1]) * 100),
        })

    return {
        "episode": episode,
        "mode": mode,
        "projected": proj,
        "actual": actual,
        "errors": pd.DataFrame(errs),
        "n_train_quarters": len(hist),
    }


def run_all(panel, macro, cashflow, income_extras,
            modes=("conditional", "macro")) -> Dict[str, Dict]:
    out = {}
    for ep in EPISODES:
        for mode in modes:
            r = replay_episode(panel, macro, cashflow, income_extras, ep, mode)
            out[f"{ep.name} [{mode}]"] = r
    return out


if __name__ == "__main__":
    from features import load_panel
    from ingest.cashflow import build_cashflow_panel, build_income_extras

    panel = load_panel()
    macro = pd.read_csv(os.path.join(ROOT, "data", "cache", "macro_quarterly.csv"),
                        index_col=0, parse_dates=True)
    cf = build_cashflow_panel()
    ie = build_income_extras()

    print("=" * 78)
    print("HISTORICAL REPLAY VALIDATION")
    print("The twin is re-fitted on pre-episode data only — no lookahead.")
    print("=" * 78)

    summary = []
    for ep in EPISODES:
        for mode in ("conditional", "macro"):
            r = replay_episode(panel, macro, cf, ie, ep, mode)
            if "error" in r:
                print(f"\n{ep.name} [{mode}]: SKIPPED — {r['error']}")
                continue
            e = r["errors"].set_index("metric")
            print(f"\n{'-'*78}\n{ep.name}  [{mode}]   "
                  f"(trained on {r['n_train_quarters']} quarters to {ep.start})")
            print(ep.description)
            print()
            print(e[["actual_final", "twin_final", "MAPE_%", "final_err_%"]]
                  .round(3).to_string())
            for m in ("revenue", "operating_income", "gross_margin", "cash"):
                if m in e.index:
                    summary.append({"episode": ep.name, "mode": mode,
                                    "metric": m,
                                    "MAPE_%": e.loc[m, "MAPE_%"]})

    print(f"\n{'='*78}\nSUMMARY (MAPE %, lower is better)\n{'='*78}")
    sdf = pd.DataFrame(summary)
    piv = sdf.pivot_table(index=["episode", "metric"], columns="mode",
                          values="MAPE_%")
    print(piv.round(2).to_string())

    # persist for the dashboard
    out = os.path.join(ROOT, "data", "cache")
    sdf.to_csv(os.path.join(out, "replay_summary.csv"), index=False)
    frames = []
    for ep in EPISODES:
        for mode in ("conditional", "macro"):
            r = replay_episode(panel, macro, cf, ie, ep, mode)
            if "error" in r:
                continue
            pr = r["projected"].add_prefix("twin_")
            ac = r["actual"].add_prefix("actual_")
            j = pr.join(ac, how="inner")
            j["episode"] = ep.name
            j["mode"] = mode
            frames.append(j.reset_index())
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            os.path.join(out, "replay_paths.csv"), index=False)
        print(f"\nSaved replay paths -> {out}/replay_paths.csv")
