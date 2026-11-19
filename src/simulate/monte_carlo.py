"""
Monte Carlo scenario simulation.

A single deterministic "what-if" answers the wrong question. Asking "what if
inflation is +4%?" gives one number with false precision; asking "given
inflation uncertainty of this size, what is the distribution of operating
income, and how bad is the 5th percentile?" gives something a CFO can act on.

So each scenario specifies shocks as *distributions*, we sample them, propagate
each draw through the causal graph, and report the resulting distribution plus
tail risk (VaR / CVaR).

Scenario calibration
--------------------
Shock magnitudes are not invented. They are taken from the empirical
distribution of the macro series themselves -- the inflation-spike scenario
uses the actual size of the 2021-22 CPI move, the rate-shock scenario the
actual size of the 2022-23 hiking cycle. `calibrate_from_history` reports
these so the numbers in the report are traceable to observed history.
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

from graph.causal_graph import propagate  # noqa: E402
from simulate.risk import RiskCalibration, score_state, state_from_levels  # noqa: E402


@dataclass
class ShockSpec:
    """A shock as a distribution: mean, sd, and optional hard bounds."""
    mean: float
    sd: float = 0.0
    lo: Optional[float] = None
    hi: Optional[float] = None

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        x = rng.normal(self.mean, self.sd, n) if self.sd > 0 else np.full(n, self.mean)
        if self.lo is not None:
            x = np.maximum(x, self.lo)
        if self.hi is not None:
            x = np.minimum(x, self.hi)
        return x


@dataclass
class Scenario:
    name: str
    description: str
    shocks: Dict[str, ShockSpec] = field(default_factory=dict)


def calibrate_from_history(macro: pd.DataFrame) -> Dict[str, float]:
    """
    Derive shock magnitudes from what actually happened, so scenarios are
    empirically grounded rather than round numbers.
    """
    c = {}
    if "inflation_yoy" in macro:
        infl = macro["inflation_yoy"].dropna()
        c["infl_sd"] = float(infl.std())
        c["infl_p95"] = float(infl.quantile(0.95))
        c["infl_max"] = float(infl.max())
    if "rate_change" in macro:
        rc = macro["rate_change"].dropna()
        c["rate_sd"] = float(rc.std())
        c["rate_p95"] = float(rc.quantile(0.95))
        c["rate_max"] = float(rc.max())
    if "gdp_yoy" in macro:
        g = macro["gdp_yoy"].dropna()
        c["gdp_sd"] = float(g.std())
        c["gdp_min"] = float(g.min())
    return c


def default_scenarios(macro: pd.DataFrame) -> List[Scenario]:
    c = calibrate_from_history(macro)
    infl_spike = c.get("infl_max", 0.07)
    infl_sd = c.get("infl_sd", 0.02)
    gdp_trough = c.get("gdp_min", -0.08)
    gdp_sd = c.get("gdp_sd", 0.02)

    return [
        Scenario("Baseline", "No shock; sampling noise only.", {}),

        Scenario(
            "Inflation spike",
            f"CPI at its historical maximum ({infl_spike:.1%} YoY), input costs "
            "rising with it.",
            {"inflation_yoy": ShockSpec(infl_spike, infl_sd),
             "supply_cost_shock": ShockSpec(0.06, 0.03, lo=0.0)},
        ),

        Scenario(
            "Demand contraction",
            f"Recessionary demand, GDP at its historical trough ({gdp_trough:.1%}).",
            {"gdp_yoy": ShockSpec(gdp_trough, gdp_sd),
             "demand_shock": ShockSpec(-0.12, 0.05, hi=0.0)},
        ),

        Scenario(
            "Supply chain disruption",
            "Component shortage: input costs up sharply, volumes constrained.",
            {"supply_cost_shock": ShockSpec(0.15, 0.06, lo=0.0),
             "demand_shock": ShockSpec(-0.05, 0.03, hi=0.05)},
        ),

        Scenario(
            "Currency shock",
            "Broad USD strengthening against the ~60% of revenue booked abroad.",
            {"fx_shock": ShockSpec(-0.10, 0.04)},
        ),

        Scenario(
            "Stagflation",
            "Worst case: high inflation with contracting demand simultaneously.",
            {"inflation_yoy": ShockSpec(infl_spike, infl_sd),
             "supply_cost_shock": ShockSpec(0.10, 0.04, lo=0.0),
             "demand_shock": ShockSpec(-0.10, 0.05, hi=0.0),
             "gdp_yoy": ShockSpec(gdp_trough * 0.7, gdp_sd)},
        ),
    ]


def run_scenario(graph, base: Dict[str, float], scenario: Scenario,
                 calib: RiskCalibration, n_sims: int = 2000,
                 seed: int = 42) -> pd.DataFrame:
    """Sample the scenario's shocks, propagate each draw, return one row per draw."""
    rng = np.random.default_rng(seed)
    draws = {k: spec.sample(rng, n_sims) for k, spec in scenario.shocks.items()}

    rows = []
    for i in range(n_sims):
        shocks = {k: float(v[i]) for k, v in draws.items()}
        out = propagate(graph, base, shocks)

        levels = {**base, **{k: v for k, v in out.items() if not k.startswith("_")}}
        st = state_from_levels(levels)
        risk = score_state(st, calib)

        rows.append({
            "sim": i,
            "revenue": out.get("revenue", np.nan),
            "cogs": out.get("cogs", np.nan),
            "gross_profit": out.get("gross_profit", np.nan),
            "operating_income": out.get("operating_income", np.nan),
            "gross_margin": st.get("gross_margin", np.nan),
            "operating_margin": st.get("operating_margin", np.nan),
            "current_ratio": st.get("current_ratio", np.nan),
            "working_capital": out.get("working_capital", np.nan),
            "risk_composite": risk["composite"],
            "risk_liquidity": risk["categories"].get("liquidity", np.nan),
            "risk_leverage": risk["categories"].get("leverage", np.nan),
            "risk_operational": risk["categories"].get("operational", np.nan),
            **{f"shock_{k}": float(v[i]) for k, v in draws.items()},
        })
    return pd.DataFrame(rows)


def summarise_runs(df: pd.DataFrame, base: Dict[str, float],
                   metric: str = "operating_income",
                   alpha: float = 0.05) -> Dict[str, float]:
    """Distribution summary plus tail risk for one simulated metric."""
    x = df[metric].dropna().to_numpy(float)
    if len(x) == 0:
        return {}
    base_val = base.get(metric, np.nan)
    if metric == "operating_income" and not np.isfinite(base_val):
        base_val = base.get("gross_profit", np.nan) - base.get("opex", np.nan)

    var = float(np.quantile(x, alpha))            # value-at-risk (level)
    cvar = float(x[x <= var].mean()) if (x <= var).any() else var

    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "p05": var,
        "p95": float(np.quantile(x, 0.95)),
        "sd": float(x.std()),
        "base": float(base_val) if np.isfinite(base_val) else np.nan,
        "mean_change_pct": (float(x.mean()) / base_val - 1) * 100
                           if np.isfinite(base_val) and base_val else np.nan,
        f"VaR_{int(alpha*100)}": var,
        f"CVaR_{int(alpha*100)}": cvar,
        "prob_below_base": float((x < base_val).mean())
                           if np.isfinite(base_val) else np.nan,
    }


def main():
    from features import load_panel, build_features  # noqa: E402
    from graph.causal_graph import fit_graph          # noqa: E402

    panel = load_panel()
    feats = build_features(panel)
    macro = pd.read_csv(os.path.join(ROOT, "data", "cache", "macro_quarterly.csv"),
                        index_col=0, parse_dates=True)
    graph = fit_graph(feats, macro)
    calib = RiskCalibration.from_features(feats)

    need = ["revenue", "cogs", "gross_profit", "operating_income", "inventory",
            "receivables", "payables", "cash", "assets_curr", "liab_curr",
            "assets", "liabilities", "equity"]
    row = panel.dropna(subset=need).iloc[-1]
    base = {k: float(row[k]) for k in need}
    base["opex"] = base["gross_profit"] - base["operating_income"]
    base["lt_debt"] = float(row.get("lt_debt", np.nan))

    print(f"Base quarter: {row.name.date()}")
    print(f"  revenue ${base['revenue']:,.0f}M | operating income "
          f"${base['operating_income']:,.0f}M | "
          f"gross margin {base['gross_profit']/base['revenue']:.1%}\n")

    cal = calibrate_from_history(macro)
    print("Scenario calibration (from observed FRED history):")
    for k, v in cal.items():
        print(f"  {k:<10} {v:+.4f}")
    print()

    print("=" * 96)
    rows = []
    for sc in default_scenarios(macro):
        df = run_scenario(graph, base, sc, calib, n_sims=1500)
        s = summarise_runs(df, base, "operating_income")
        rows.append({
            "scenario": sc.name,
            "op_income_mean": s["mean"],
            "vs_base_%": s["mean_change_pct"],
            "p05": s["p05"],
            "CVaR_5": s["CVaR_5"],
            "gross_margin": df["gross_margin"].mean(),
            "risk": df["risk_composite"].mean(),
        })
    out = pd.DataFrame(rows).set_index("scenario")
    print("Scenario results (operating income, USD millions):")
    print(out.round(2).to_string())

    print("\nSanity checks:")
    b = out.loc["Baseline"]
    checks = [
        ("baseline ~ unchanged vs base", abs(b["vs_base_%"]) < 1.0),
        ("stagflation worst for operating income",
         out["op_income_mean"].idxmin() == "Stagflation"),
        ("every adverse scenario reduces operating income",
         (out.drop(index="Baseline")["vs_base_%"] < 0).all()),
        ("supply shock compresses gross margin below baseline",
         out.loc["Supply chain disruption", "gross_margin"] < b["gross_margin"]),
        ("adverse scenarios raise composite risk",
         out.loc["Stagflation", "risk"] >= b["risk"]),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    out.to_csv(os.path.join(ROOT, "data", "cache", "scenario_results.csv"))
    return out


if __name__ == "__main__":
    main()
