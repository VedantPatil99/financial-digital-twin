"""
Tests for the three-statement engine, temporal dynamics and decision layer.

The balance-sheet checks matter most: the engine's whole claim is that the
sheet ties by construction, so any drift is a bug rather than rounding. These
assert to $0.01M across long horizons and extreme shocks.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import load_panel, build_features                    # noqa: E402
from graph.causal_graph import fit_graph                           # noqa: E402
from ingest.cashflow import build_cashflow_panel, build_income_extras  # noqa: E402
from twin.statements import calibrate, opening_state, step         # noqa: E402
from twin.dynamics import (estimate_lags, baseline_path, ShockPath,  # noqa: E402
                           simulate_trajectory, cash_runway)
from twin.decisions import TwinRunner, goal_seek, tornado, LEVER_BY_KEY  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        FAILURES.append(name)


def main():
    panel = load_panel()
    feats = build_features(panel)
    macro = pd.read_csv(os.path.join(ROOT, "data", "cache",
                                     "macro_quarterly.csv"),
                        index_col=0, parse_dates=True)
    cf = build_cashflow_panel()
    ie = build_income_extras()
    graph = fit_graph(feats, macro)
    lags = estimate_lags(feats, macro)
    params = calibrate(panel, cf, ie)
    s0 = opening_state(panel, cf, ie)

    print("Opening state")
    check("opening balance sheet ties", abs(s0.balance_error) < 1e-6,
          f"(error ${s0.balance_error:,.9f}M)")
    check("opening cash positive", s0.cash > 0, f"(${s0.cash:,.0f}M)")
    check("PP&E tracked separately", s0.ppe > 0, f"(${s0.ppe:,.0f}M)")

    print("\nPolicy calibration")
    check("effective tax rate in 5-40%",
          0.05 <= params.effective_tax_rate <= 0.40,
          f"({params.effective_tax_rate:.1%})")
    check("capex intensity in 0-12%",
          0.0 <= params.capex_pct_revenue <= 0.12,
          f"({params.capex_pct_revenue:.2%})")
    check("quarterly depreciation rate in 0.5-25%",
          0.005 <= params.depreciation_rate_ppe <= 0.25,
          f"({params.depreciation_rate_ppe:.2%})")

    bl = baseline_path(panel, 12, cashflow=cf)
    print("\nBaseline path")
    check("baseline has seasonality (max/min > 1.2)",
          bl["revenue"].max() / bl["revenue"].min() > 1.2,
          f"({bl['revenue'].max()/bl['revenue'].min():.2f}x)")
    check("baseline revenue positive", (bl["revenue"] > 0).all())

    print("\nBalance sheet integrity across scenarios")
    scenarios = {
        "no shock": {},
        "severe demand": {"demand_shock": ShockPath(-0.35, 0.9)},
        "severe cost": {"supply_cost_shock": ShockPath(0.40, 0.9)},
        "combined extreme": {"demand_shock": ShockPath(-0.30, 0.95),
                             "supply_cost_shock": ShockPath(0.30, 0.95),
                             "inflation_yoy": ShockPath(0.09, 0.95)},
        "positive shock": {"demand_shock": ShockPath(0.25, 0.9)},
    }
    for nm, sh in scenarios.items():
        t = simulate_trajectory(graph, s0, params, sh, lags, horizon=12,
                                baseline=bl)
        err = t["balance_error"].abs().max()
        check(f"balance ties under '{nm}'", err < 0.01,
              f"(max ${err:,.9f}M over 12 quarters)")

    print("\nAccounting identities within the projection")
    t = simulate_trajectory(graph, s0, params, scenarios["combined extreme"],
                            lags, horizon=8, baseline=bl)
    check("revenue - cogs = gross profit",
          (t["revenue"] - t["cogs"] - t["gross_profit"]).abs().max() < 0.01)
    check("gross profit - opex_total = operating income",
          (t["gross_profit"] - t["opex_total"]
           - t["operating_income"]).abs().max() < 0.01)
    check("pretax - tax = net income",
          (t["pretax_income"] - t["tax"] - t["net_income"]).abs().max() < 0.01)
    check("ocf - capex = fcf",
          (t["ocf"] - t["capex"] - t["fcf"]).abs().max() < 0.01)

    print("\nEconomic direction")
    base = simulate_trajectory(graph, s0, params, {}, lags, horizon=8,
                               baseline=bl)
    dem = simulate_trajectory(graph, s0, params,
                              {"demand_shock": ShockPath(-0.20, 0.9)}, lags,
                              horizon=8, baseline=bl)
    cost = simulate_trajectory(graph, s0, params,
                               {"supply_cost_shock": ShockPath(0.20, 0.9)},
                               lags, horizon=8, baseline=bl)
    check("demand shock lowers revenue",
          dem["revenue"].sum() < base["revenue"].sum())
    check("demand shock lowers operating income",
          dem["operating_income"].sum() < base["operating_income"].sum())
    check("cost shock compresses gross margin",
          cost["gross_margin"].mean() < base["gross_margin"].mean(),
          f"({base['gross_margin'].mean():.1%} -> {cost['gross_margin'].mean():.1%})")
    check("cost shock barely moves revenue",
          abs(cost["revenue"].sum() / base["revenue"].sum() - 1) < 0.01)
    check("demand shock hurts margin less than cost shock does",
          dem["gross_margin"].mean() > cost["gross_margin"].mean(),
          "(a volume shock is not a margin shock)")

    print("\nInvestment loop (capex -> PP&E -> D&A)")
    hi = simulate_trajectory(graph, s0, params.__class__(
        **{**params.__dict__, "capex_pct_revenue": 0.10}), {}, lags,
        horizon=8, baseline=bl)
    check("higher capex raises PP&E", hi["ppe"].iloc[-1] > base["ppe"].iloc[-1],
          f"(${hi['ppe'].iloc[-1]:,.0f}M vs ${base['ppe'].iloc[-1]:,.0f}M)")
    check("higher capex raises D&A", hi["dna"].sum() > base["dna"].sum())
    check("higher capex lowers operating income",
          hi["operating_income"].sum() < base["operating_income"].sum())
    check("higher capex lowers cash",
          hi["cash"].iloc[-1] < base["cash"].iloc[-1])

    print("\nFinancing feedback")
    dbt = params.__class__(**{**params.__dict__, "interest_rate_on_debt": 0.10})
    hid = simulate_trajectory(graph, s0, dbt, {}, lags, horizon=8, baseline=bl)
    check("higher cost of debt lowers net income",
          hid["net_income"].sum() < base["net_income"].sum())
    check("nonoperating income falls with rate",
          hid["nonoperating_income"].sum() < base["nonoperating_income"].sum())

    print("\nDecision layer")
    runner = TwinRunner(graph, s0, params, lags, bl, horizon=8)
    r = goal_seek(runner, "buyback_pct_ocf", "cash_final",
                  float(base["cash"].iloc[-1]) * 1.5)
    check("goal-seek returns a result", r is not None)
    if r.feasible:
        check("goal-seek solution within lever bounds",
              LEVER_BY_KEY["buyback_pct_ocf"].lo <= r.solution
              <= LEVER_BY_KEY["buyback_pct_ocf"].hi,
              f"(solved {r.solution:.3f})")
        check("goal-seek achieves its target",
              abs(r.achieved / (float(base['cash'].iloc[-1]) * 1.5) - 1) < 0.02,
              f"(achieved ${r.achieved:,.0f}M)")
    else:
        check("infeasible goal reported with a reason", bool(r.message))

    trn = tornado(runner, "net_income_total")
    check("tornado ranks every factor", len(trn) >= 8, f"({len(trn)} factors)")
    check("tornado sorted by swing",
          (trn["swing"].diff().dropna() <= 1e-6).all())
    check("shocks dominate net income",
          trn.iloc[0]["kind"] == "shock",
          f"(top factor: {trn.iloc[0]['factor']})")

    trc = tornado(runner, "cash_final")
    check("levers dominate final cash",
          trc.iloc[0]["kind"] == "lever",
          f"(top factor: {trc.iloc[0]['factor']})")

    print(f"\n{len(FAILURES)} failure(s): {FAILURES}" if FAILURES
          else "\nAll three-statement tests passed.")
    return 1 if FAILURES else 0


def test_all_checks_pass():
    """Entry point for pytest: every check above must pass."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
