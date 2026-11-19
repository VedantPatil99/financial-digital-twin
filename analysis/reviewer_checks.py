"""
Reviewer-requested checks for the IEEE paper draft.

Runs on the project's own cached data and imports the project's own modules
unchanged. Nothing in src/ is modified. Every number written to analysis/out/
comes from this script.

    python analysis/reviewer_checks.py

Checks
------
1. HAC (Newey-West) standard errors for every behavioral elasticity, next to
   the plain OLS standard errors the pipeline reports.
2. Diebold-Mariano tests (Harvey-Leybourne-Newbold small-sample correction)
   of each forecasting model against SeasonalNaive, plus MASE, for revenue,
   COGS and operating income.
3. Historical replay against simple benchmarks: a no-shock run of the same
   engine (macro mode) and constant-margin / flat-cash rules (conditional).
4. Plug ablation: a conventional percent-of-sales model with cash as the
   balancing plug, driven by the twin's own P&L and working-capital paths,
   compared with the twin's cash-flow-derived cash.
5. Gross profit estimated as its own behavioral edge, compared with the
   identity revenue - COGS under the six scenario shocks.
6. Size of the one offsetting entry left in the statement engine
   (other current assets move with other current liabilities).
7. A worked decision example under a stagflation shock path.
8. A controlled level-vs-residual target test with identical GBM settings.
"""
from __future__ import annotations

import os
import sys
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

from features import load_panel, build_features                       # noqa: E402
from graph.causal_graph import (fit_graph, BEHAVIORAL, _series_for,    # noqa: E402
                                propagate, PRIOR_ELASTICITIES)
from ingest.cashflow import build_cashflow_panel, build_income_extras  # noqa: E402
from twin.statements import calibrate, opening_state                    # noqa: E402
from twin.dynamics import (estimate_lags, baseline_path, ShockPath,    # noqa: E402
                           simulate_trajectory)
from twin.decisions import TwinRunner, goal_seek, tornado              # noqa: E402
from twin import replay as R                                           # noqa: E402
from simulate.monte_carlo import default_scenarios                     # noqa: E402

panel = load_panel()
feats = build_features(panel)
macro = pd.read_csv(os.path.join(ROOT, "data", "cache", "macro_quarterly.csv"),
                    index_col=0, parse_dates=True)
cf = build_cashflow_panel()
ie = build_income_extras()
graph = fit_graph(feats, macro)
RESULTS = {}


def save(name, df):
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"\n### {name}\n" + df.to_string(index=False))


# ---------------------------------------------------------------------------
# 1. HAC standard errors
# ---------------------------------------------------------------------------
def hac_elasticities(maxlags=4):
    rows = []
    for child, parents in BEHAVIORAL.items():
        y_s = _series_for(child, feats, macro, "yoy")
        observed = [p for p in parents if _series_for(p, feats, macro, "yoy") is not None]
        df = pd.DataFrame({p: _series_for(p, feats, macro, "yoy") for p in observed})
        df["__y"] = y_s
        fq = feats["fiscal_q"].reindex(df.index)
        for q in (2, 3, 4):
            df[f"__q{q}"] = (fq == q).astype(float)
        df = df.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
        X = sm.add_constant(df[observed + [c for c in df.columns if c.startswith("__q")]])
        ols = sm.OLS(df["__y"], X).fit()
        hac = sm.OLS(df["__y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
        for p in observed:
            rows.append({
                "parent": p, "child": child, "n": int(ols.nobs),
                "elasticity": ols.params[p],
                "se_ols": ols.bse[p], "t_ols": ols.tvalues[p],
                "se_hac": hac.bse[p], "t_hac": hac.tvalues[p],
                "p_hac": hac.pvalues[p], "r2": ols.rsquared,
            })
    out = pd.DataFrame(rows)
    save("hac_elasticities", out.round(4))
    return out


# ---------------------------------------------------------------------------
# 2. Diebold-Mariano + MASE
# ---------------------------------------------------------------------------
def dm_test(e_model, e_base, loss="ape", actual=None, h=1):
    if loss == "ape":
        d = np.abs(e_model / actual) - np.abs(e_base / actual)
    else:
        d = e_model ** 2 - e_base ** 2
    T = len(d)
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    var = gamma0 / T                      # h = 1: no autocovariance terms
    dm = dbar / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return stat, p


def forecast_tests():
    rows = []
    for target in ("revenue", "cogs", "operating_income"):
        bt = pd.read_csv(os.path.join(ROOT, "data", "cache", f"backtest_{target}.csv"),
                         parse_dates=["period"])
        y = panel[target].dropna()
        first_oos = bt["period"].min()
        insample = y[y.index < first_oos]
        scale = float(np.mean(np.abs(insample.diff(4).dropna())))
        base = bt[bt.model == "SeasonalNaive"].set_index("period").sort_index()
        for m, g in bt.groupby("model"):
            g = g.set_index("period").sort_index().loc[base.index]
            e = (g["pred"] - g["actual"]).to_numpy()
            eb = (base["pred"] - base["actual"]).to_numpy()
            a = g["actual"].to_numpy()
            row = {"target": target, "model": m, "n": len(g),
                   "MAPE": float(np.mean(np.abs(e / a)) * 100),
                   "MASE": float(np.mean(np.abs(e)) / scale)}
            if m != "SeasonalNaive":
                s1, p1 = dm_test(e, eb, "ape", a)
                s2, p2 = dm_test(e, eb, "se")
                row.update({"DM_ape": s1, "p_ape": p1, "DM_se": s2, "p_se": p2})
            rows.append(row)
    out = pd.DataFrame(rows).sort_values(["target", "MAPE"])
    save("forecast_dm_mase", out.round(4))
    return out


# ---------------------------------------------------------------------------
# 3. Replay against simple benchmarks
# ---------------------------------------------------------------------------
def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    m = np.isfinite(a) & np.isfinite(p) & (a != 0)
    return float(np.mean(np.abs((p[m] - a[m]) / a[m])) * 100)


def replay_benchmarks():
    rows = []
    for ep in R.EPISODES:
        start = pd.Timestamp(ep.start)
        # (a) twin, both modes, exactly as the pipeline runs them
        res = {mode: R.replay_episode(panel, macro, cf, ie, ep, mode)
               for mode in ("conditional", "macro")}
        # (b) no-shock run of the same engine: future macro set to the
        #     pre-episode means the macro mode subtracts, so every shock is 0
        mz = macro.copy()
        hist = macro[macro.index <= start]
        fut = mz.index > start
        mz.loc[fut, "inflation_yoy"] = float(hist["inflation_yoy"].tail(8).mean())
        mz.loc[fut, "gdp_yoy"] = float(hist["gdp_yoy"].tail(8).mean())
        noshock = R.replay_episode(panel, mz, cf, ie, ep, "macro")

        actual = res["conditional"]["actual"]
        hist_p = panel[panel.index <= start].dropna(
            subset=["revenue", "gross_profit", "operating_income"]).tail(8)
        gm_rule = float((hist_p["gross_profit"] / hist_p["revenue"]).median())
        om_rule = float((hist_p["operating_income"] / hist_p["revenue"]).median())
        last_cash = float(panel.loc[panel.index <= start, "cash"].dropna().iloc[-1])

        def add(model, mode, proj):
            rows.append({
                "episode": ep.name, "mode": mode, "model": model,
                "quarters": len(actual),
                "revenue_MAPE": mape(actual["revenue"], proj["revenue"]),
                "op_income_MAPE": mape(actual["operating_income"], proj["operating_income"]),
                "gross_margin_MAE_pp": float(np.nanmean(np.abs(
                    proj["gross_margin"].to_numpy(float) - actual["gross_margin"].to_numpy(float))) * 100),
                "cash_MAPE": mape(actual["cash"], proj["cash"]),
            })

        add("twin", "conditional", res["conditional"]["projected"])
        add("twin", "macro", res["macro"]["projected"])
        add("no-shock engine run", "macro", noshock["projected"])
        rule = pd.DataFrame({
            "revenue": actual["revenue"].to_numpy(float),
            "operating_income": actual["revenue"].to_numpy(float) * om_rule,
            "gross_margin": np.full(len(actual), gm_rule),
            "cash": np.full(len(actual), last_cash),
        }, index=actual.index)
        add("constant-margin, flat-cash rule", "conditional", rule)
    out = pd.DataFrame(rows)
    save("replay_benchmarks", out.round(3))
    return out


# ---------------------------------------------------------------------------
# shared: stagflation and other scenario means as shock paths
# ---------------------------------------------------------------------------
def scenario_means():
    out = {}
    for sc in default_scenarios(macro):
        out[sc.name] = {k: v.mean for k, v in sc.shocks.items()}
    return out


SC = scenario_means()
lags = estimate_lags(feats, macro)
params = calibrate(panel, cf, ie)
s0 = opening_state(panel, cf, ie)
H = 8
bl = baseline_path(panel, H, cashflow=cf)
runner = TwinRunner(graph, s0, params, lags, bl, horizon=H)


def paths(name, persistence=0.85):
    return {k: ShockPath(v, persistence=persistence) for k, v in SC[name].items()}


# ---------------------------------------------------------------------------
# 4. Plug ablation and 6. offsetting entry
# ---------------------------------------------------------------------------
def plug_ablation():
    rows = []
    rev0 = s0.revenue
    share = {
        "ppe": s0.ppe / rev0,
        "other_curr_assets": s0.other_curr_assets / rev0,
        "other_noncurr_assets": s0.other_noncurr_assets / rev0,
        "other_curr_liab": s0.other_curr_liab / rev0,
        "other_noncurr_liab": s0.other_noncurr_liab / rev0,
    }
    for name in SC:
        tr = runner.run(paths(name))
        prev_cash_cf = s0.cash
        prev_eq = s0.equity
        prev_ocl = s0.other_curr_liab
        max_gap, max_offset, cum_offset = 0.0, 0.0, 0.0
        # variant 2: only working-capital lines scale with revenue; PP&E rolls
        # forward with capex - D&A; non-current lines are held flat
        ppe_v2, max_gap_v2 = s0.ppe, 0.0
        for _, r in tr.iterrows():
            rev = r["revenue"]
            ppe_v2 = ppe_v2 + r["capex"] - r["dna"]
            nca2 = (r["receivables"] + r["inventory"] + share["other_curr_assets"] * rev
                    + ppe_v2 + s0.other_noncurr_assets)
            liab2 = (r["payables"] + share["other_curr_liab"] * rev
                     + s0.debt + s0.other_noncurr_liab)
            eq2 = prev_eq + r["net_income"] - r["dividends"] - r["buybacks"]
            cash_cf2 = prev_cash_cf + r["ocf"] - r["capex"] - r["dividends"] - r["buybacks"]
            max_gap_v2 = max(max_gap_v2, abs(liab2 + eq2 - nca2 - cash_cf2))
            # percent-of-sales balance sheet, cash as the plug
            nca = (r["receivables"] + r["inventory"]
                   + share["ppe"] * rev + share["other_curr_assets"] * rev
                   + share["other_noncurr_assets"] * rev)
            eq = prev_eq + r["net_income"] - r["dividends"] - r["buybacks"]
            liab = (r["payables"] + share["other_curr_liab"] * rev
                    + s0.debt + share["other_noncurr_liab"] * rev)
            cash_plug = liab + eq - nca
            # the same model's own cash-flow statement (indirect method)
            cash_cf = prev_cash_cf + r["ocf"] - r["capex"] - r["dividends"] - r["buybacks"]
            gap = cash_plug - cash_cf
            max_gap = max(max_gap, abs(gap))
            # twin: other current assets move one-for-one with other current liabilities
            ocl = params.other_curr_liab_pct_revenue * rev
            off = abs(ocl - prev_ocl)
            max_offset = max(max_offset, off)
            cum_offset += off
            prev_ocl, prev_cash_cf, prev_eq = ocl, cash_cf, eq
        rows.append({
            "scenario": name,
            "twin_max_balance_error": float(tr["balance_error"].abs().max()),
            "plug_model_max_gap_vs_own_cashflow": max_gap,
            "plug_gap_pct_of_twin_final_cash": max_gap / float(tr["cash"].iloc[-1]) * 100,
            "plug_model_v2_max_gap": max_gap_v2,
            "plug_v2_gap_pct_of_twin_final_cash": max_gap_v2 / float(tr["cash"].iloc[-1]) * 100,
            "twin_min_current_ratio": float(tr["current_ratio"].min()),
            "twin_max_current_ratio": float(tr["current_ratio"].max()),
            "twin_offset_entry_max_per_quarter": max_offset,
            "twin_offset_entry_cumulative": cum_offset,
            "twin_final_cash": float(tr["cash"].iloc[-1]),
        })
    out = pd.DataFrame(rows)
    save("plug_ablation", out.round(2))
    return out


# ---------------------------------------------------------------------------
# 5. Gross profit as its own behavioral edge
# ---------------------------------------------------------------------------
def gross_profit_ablation():
    gp = np.log(panel["gross_profit"]).diff(4)
    df = pd.DataFrame({"y": gp, "rev": feats["dlog4_revenue"],
                       "infl": macro["inflation_yoy"]})
    fq = feats["fiscal_q"]
    for q in (2, 3, 4):
        df[f"q{q}"] = (fq == q).astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    fit = sm.OLS(df["y"], sm.add_constant(df[["rev", "infl", "q2", "q3", "q4"]])).fit()
    b_rev, b_inf = fit.params["rev"], fit.params["infl"]

    need = ["revenue", "cogs", "gross_profit", "operating_income", "inventory",
            "receivables", "payables", "cash", "assets_curr", "liab_curr",
            "assets", "liabilities", "equity"]
    row = panel.dropna(subset=need).iloc[-1]
    base = {k: float(row[k]) for k in need}
    base["opex"] = base["gross_profit"] - base["operating_income"]
    rows = []
    for name, shocks in SC.items():
        out = propagate(graph, base, shocks)
        d = out["_dlog"]
        gp_direct = base["gross_profit"] * np.exp(b_rev * d.get("revenue", 0.0)
                                                  + b_inf * d.get("inflation_yoy", 0.0))
        gp_ident = out["revenue"] - out["cogs"]
        rows.append({"scenario": name, "gp_identity": gp_ident, "gp_direct": gp_direct,
                     "gap": gp_direct - gp_ident,
                     "gap_pct_of_gp": (gp_direct - gp_ident) / gp_ident * 100})
    out = pd.DataFrame(rows)
    RESULTS["gp_edge"] = {"b_rev": float(b_rev), "b_infl": float(b_inf),
                          "n": int(fit.nobs), "r2": float(fit.rsquared)}
    print("\nGross-profit edge:", RESULTS["gp_edge"])
    save("gross_profit_ablation", out.round(2))
    return out


# ---------------------------------------------------------------------------
# 7. Decision example
# ---------------------------------------------------------------------------
def decision_example():
    base_tr = runner.run({})
    stag = paths("Stagflation")
    st_tr = runner.run(stag)
    target = float(s0.cash)       # keep the lowest cash balance at or above opening cash
    gs = goal_seek(runner, "buyback_pct_ocf", "cash_min", target, shocks=stag)
    trn = tornado(runner, "cash_min", base_shocks=stag)
    info = {
        "horizon_quarters": H,
        "opening_cash": float(s0.cash),
        "calibrated_buyback_pct_ocf": float(params.buyback_pct_ocf),
        "baseline_cash_min": float(base_tr["cash"].min()),
        "stagflation_cash_min": float(st_tr["cash"].min()),
        "stagflation_op_income_total": float(st_tr["operating_income"].sum()),
        "baseline_op_income_total": float(base_tr["operating_income"].sum()),
        "goal_seek_feasible": bool(gs.feasible),
        "goal_seek_solution": gs.solution,
        "goal_seek_achieved": gs.achieved,
        "goal_seek_message": gs.message,
        "stagflation_shocks": SC["Stagflation"],
        "persistence": 0.85,
    }
    RESULTS["decision"] = info
    print("\nDecision example:", json.dumps(info, indent=2, default=float))
    save("decision_tornado_cash_min_stagflation",
         trn[["factor", "kind", "delta", "low", "high", "swing", "share_of_swing"]].round(4))
    return info


def controlled_target_test():
    """
    The pipeline's GradientBoost (level) uses 300 trees and ResidualGBM uses
    200, so comparing them mixes a target change with a hyperparameter change.
    Re-run both targets with identical estimator settings.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from forecast.models import GradientBoost, ResidualModel
    from forecast.backtest import walk_forward

    rows = []
    for n_est in (200, 300):
        def mk(n=n_est):
            return GradientBoostingRegressor(n_estimators=n, max_depth=2, learning_rate=0.05,
                                             subsample=0.9, random_state=0)

        class Level(GradientBoost):
            name = f"GBM_level_{n_est}"

            def __init__(self):
                self.model = mk()

        models = [Level(), ResidualModel(mk(), f"GBM_residual_{n_est}", scale=False)]
        for target in ("revenue", "cogs", "operating_income"):
            y = panel[target].dropna()
            bt = walk_forward(y, macro, panel["fiscal_q"], models=models)
            naive = pd.read_csv(os.path.join(ROOT, "data", "cache", f"backtest_{target}.csv"))
            nb = naive[naive.model == "SeasonalNaive"]
            base_mape = float(np.mean(np.abs((nb.pred - nb.actual) / nb.actual)) * 100)
            for m, g in bt.groupby("model"):
                mp = float(np.mean(np.abs((g.pred - g.actual) / g.actual)) * 100)
                rows.append({"target": target, "model": m, "n_estimators": n_est,
                             "MAPE": mp, "skill_vs_naive_%": (base_mape - mp) / base_mape * 100})
    out = pd.DataFrame(rows)
    save("controlled_target_test", out.round(3))
    return out


def opening_residuals():
    info = {"other_curr_assets": s0.other_curr_assets,
            "other_noncurr_assets": s0.other_noncurr_assets,
            "other_curr_liab": s0.other_curr_liab,
            "other_noncurr_liab": s0.other_noncurr_liab,
            "assets": s0.assets, "balance_error": s0.balance_error,
            "period": str(s0.period.date())}
    RESULTS["opening_residuals"] = info
    print("\nOpening residual line items:", info)


if __name__ == "__main__":
    hac_elasticities()
    forecast_tests()
    replay_benchmarks()
    plug_ablation()
    gross_profit_ablation()
    controlled_target_test()
    opening_residuals()
    decision_example()
    RESULTS["lags"] = {f"{p}->{c}": int(L) for (p, c), L in lags.items()}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=float)
    print("\nWrote", OUT)
