"""Smoke + sanity tests for causal propagation and the accounting identities."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import load_panel, build_features          # noqa: E402
from graph.causal_graph import fit_graph, propagate      # noqa: E402


def latest_state(panel):
    """Most recent quarter with a complete set of the fields we propagate."""
    need = ["revenue", "cogs", "gross_profit", "inventory", "receivables",
            "payables", "cash", "assets_curr", "liab_curr"]
    df = panel.dropna(subset=need)
    row = df.iloc[-1]
    base = {k: float(row[k]) for k in need}
    base["opex"] = float(row["gross_profit"] - row["operating_income"])
    return row.name, base


def main():
    panel = load_panel()
    feats = build_features(panel)
    macro = pd.read_csv(os.path.join(ROOT, "data", "cache", "macro_quarterly.csv"),
                        index_col=0, parse_dates=True)
    g = fit_graph(feats, macro)

    when, base = latest_state(panel)
    print(f"Base state @ {when.date()}  revenue=${base['revenue']:,.0f}M "
          f"cogs=${base['cogs']:,.0f}M\n")

    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    # 1. null shock must be a no-op
    out = propagate(g, base, {})
    check("zero shock leaves revenue unchanged",
          abs(out["revenue"] - base["revenue"]) < 1e-6)
    check("zero shock leaves gross profit consistent",
          abs((out["revenue"] - out["cogs"]) - out["gross_profit"]) < 1e-6)

    # 2. demand shock direction and magnitude
    out = propagate(g, base, {"demand_shock": 0.10})
    rev_chg = out["revenue"] / base["revenue"] - 1
    cogs_chg = out["cogs"] / base["cogs"] - 1
    check("+10% demand raises revenue ~10%", abs(rev_chg - 0.10) < 0.005,
          f"(got {rev_chg:+.2%})")
    check("COGS rises less than revenue (operating leverage)",
          cogs_chg < rev_chg, f"(cogs {cogs_chg:+.2%} vs rev {rev_chg:+.2%})")
    check("gross profit rises on demand shock",
          out["gross_profit"] > base["gross_profit"])

    # 3. identity must survive an arbitrary shock
    out = propagate(g, base, {"demand_shock": -0.15, "supply_cost_shock": 0.08,
                              "inflation_yoy": 0.04})
    check("identity holds post-shock: revenue - cogs = gross_profit",
          abs((out["revenue"] - out["cogs"]) - out["gross_profit"]) < 1e-6)
    check("identity holds post-shock: gp - opex = operating income",
          abs((out["gross_profit"] - out["opex"]) - out["operating_income"]) < 1e-6)

    # 4. adverse scenario must reduce profitability
    base_om = base["gross_profit"] - base["opex"]
    check("adverse scenario cuts operating income",
          out["operating_income"] < base_om,
          f"(${out['operating_income']:,.0f}M vs ${base_om:,.0f}M)")

    # 5. supply cost shock hits margin specifically
    out_s = propagate(g, base, {"supply_cost_shock": 0.10})
    m0 = base["gross_profit"] / base["revenue"]
    m1 = out_s["gross_profit"] / out_s["revenue"]
    check("+10% input cost compresses gross margin", m1 < m0,
          f"({m0:.1%} -> {m1:.1%})")

    # 6. monotonicity: deeper demand shock => lower revenue
    revs = [propagate(g, base, {"demand_shock": d})["revenue"]
            for d in (-0.20, -0.10, 0.0, 0.10)]
    check("revenue monotonic in demand shock",
          all(np.diff(revs) > 0), f"({[f'{r:,.0f}' for r in revs]})")

    print(f"\n{len(failures)} failure(s)" if failures else "\nAll propagation tests passed.")
    return 1 if failures else 0


def test_all_checks_pass():
    """Entry point for pytest: every check above must pass."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
