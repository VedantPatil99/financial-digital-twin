"""
Figures for the paper, drawn from the project's own outputs.

    python analysis/make_figures.py

fig_dag.png         the causal graph as specified in src/graph/causal_graph.py,
                    with fitted elasticities and estimated lags
fig_elasticity.png  expanding-window elasticity estimates (data/cache/elasticity_history.csv)
fig_forecast.png    walk-forward revenue forecasts (data/cache/backtest_revenue.csv)
fig_mc.png          Monte Carlo distribution of operating income, stagflation
fig_replay.png      COVID-19 conditional replay: filed vs. twin vs. simple rule
fig_tornado.png     tornado on the lowest cash balance under stagflation
                    (analysis/out/decision_tornado_cash_min_stagflation.csv)

Colours: a colour-blind-safe three-hue set (blue, orange, aqua) plus black/grey;
every series also differs in line style or marker so the figures read in greyscale.
"""
from __future__ import annotations

import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "analysis", "out")
os.makedirs(OUT, exist_ok=True)

from features import load_panel, build_features                        # noqa: E402
from graph.causal_graph import fit_graph, BEHAVIORAL                    # noqa: E402
from twin.dynamics import estimate_lags                                 # noqa: E402
from simulate.monte_carlo import default_scenarios, run_scenario        # noqa: E402
from simulate.risk import RiskCalibration                               # noqa: E402

FONT = "Liberation Serif"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e3e3e0"
plt.rcParams.update({"font.family": FONT, "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6})

panel = load_panel()
feats = build_features(panel)
macro = pd.read_csv(os.path.join(ROOT, "data", "cache", "macro_quarterly.csv"),
                    index_col=0, parse_dates=True)
graph = fit_graph(feats, macro)
lags = estimate_lags(feats, macro)


# ---------------------------------------------------------------- DAG -----
def fig_dag():
    label = {"gdp_yoy": "GDP growth", "inflation_yoy": "CPI inflation",
             "demand_shock": "demand shock*", "supply_cost_shock": "supply-cost shock*",
             "fx_shock": "FX shock*", "revenue": "Revenue", "cogs": "COGS",
             "opex": "Opex", "inventory": "Inventory", "receivables": "Receivables",
             "payables": "Payables", "gross_profit": "Gross profit",
             "operating_income": "Operating income"}
    L = ['digraph G {', 'rankdir=TB; nodesep=0.12; ranksep=0.28; dpi=400;',
         'node [fontname="Liberation Serif", fontsize=14, shape=box, style="rounded", height=0.3, margin="0.06,0.03"];',
         'edge [fontname="Liberation Serif", fontsize=12.5, arrowsize=0.6];']
    for n in ("gdp_yoy", "inflation_yoy", "demand_shock", "supply_cost_shock", "fx_shock"):
        fill = "#fde3d8" if n.endswith("shock") else "#dbe8f8"
        L.append(f'{n} [label="{label[n]}", style="rounded,filled", fillcolor="{fill}", color="#3b4a5a"];')
    L.append('{rank=same; gdp_yoy; inflation_yoy; demand_shock; fx_shock; supply_cost_shock;}')
    for n in ("gross_profit", "operating_income"):
        L.append(f'{n} [label="{label[n]}", peripheries=2, style="rounded,filled", fillcolor="#d7f2e8", color="#3b4a5a"];')
    for n in BEHAVIORAL:
        L.append(f'{n} [label="{label[n]}", color="{BLUE}", penwidth=1.4];')
    for e in graph.edges:
        lag = lags.get((e.parent, e.child), 0)
        if e.kind == "prior":
            L.append(f'{e.parent} -> {e.child} [style=dashed, color="{ORANGE}", fontcolor="#9a3a12", label=" {e.coef:.2f}"];')
        else:
            L.append(f'{e.parent} -> {e.child} [color="{BLUE}", fontcolor="#174a8a", label=" {e.coef:.2f}' + (f' (L{lag})' if lag else '') + '"];')
    for a, b in (("revenue", "gross_profit"), ("cogs", "gross_profit"),
                 ("gross_profit", "operating_income"), ("opex", "operating_income")):
        L.append(f'{a} -> {b} [style=dotted, arrowhead=empty, color="{MUTED}"];')
    L.append('}')
    dot = "\n".join(L)
    path = os.path.join(OUT, "fig_dag.dot")
    open(path, "w").write(dot)
    subprocess.run(["dot", "-Tpng", path, "-o", os.path.join(OUT, "fig_dag.png")], check=True)


# ------------------------------------------------------------- replay -----
def fig_replay():
    rp = pd.read_csv(os.path.join(ROOT, "data", "cache", "replay_paths.csv"), parse_dates=["period"])
    ep = rp[(rp.episode == "COVID-19") & (rp["mode"] == "conditional")].sort_values("period")
    start = pd.Timestamp("2019-12-28")
    hist = panel[panel.index <= start].dropna(subset=["revenue", "operating_income"]).tail(8)
    om = float((hist["operating_income"] / hist["revenue"]).median())
    last_cash = float(panel.loc[panel.index <= start, "cash"].dropna().iloc[-1])
    x = ep["period"].dt.strftime("%b %y")
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.5), sharex=True)
    for ax, col, rule, title in (
            (axes[0], "operating_income", ep["actual_revenue"] * om, "Operating income ($B)"),
            (axes[1], "cash", pd.Series(last_cash, index=ep.index), "Cash ($B)")):
        ax.plot(x, ep[f"actual_{col}"] / 1e3, color=INK, lw=1.6, marker="o", ms=3.2, label="Filed")
        ax.plot(x, ep[f"twin_{col}"] / 1e3, color=BLUE, lw=1.5, ls="--", marker="s", ms=3.0, label="Twin (conditional)")
        ax.plot(x, rule / 1e3, color=ORANGE, lw=1.4, ls=":", marker="^", ms=3.0, label="Simple rule")
        ax.set_title(title, fontsize=8, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.8, loc="upper left", ncol=1)
    axes[1].tick_params(axis="x", labelsize=6.8)
    fig.tight_layout(pad=0.3, h_pad=0.6)
    fig.savefig(os.path.join(OUT, "fig_replay.png"), dpi=400)


# ----------------------------------------------------------------- MC -----
def fig_mc():
    calib = RiskCalibration.from_features(feats)
    need = ["revenue", "cogs", "gross_profit", "operating_income", "inventory",
            "receivables", "payables", "cash", "assets_curr", "liab_curr",
            "assets", "liabilities", "equity"]
    row = panel.dropna(subset=need).iloc[-1]
    base = {k: float(row[k]) for k in need}
    base["opex"] = base["gross_profit"] - base["operating_income"]
    base["lt_debt"] = float(row.get("lt_debt", np.nan))
    sc = [s for s in default_scenarios(macro) if s.name == "Stagflation"][0]
    df = run_scenario(graph, base, sc, calib, n_sims=1500)
    x = df["operating_income"].to_numpy() / 1e3
    var = np.quantile(x, 0.05)
    cvar = x[x <= var].mean()
    fig, ax = plt.subplots(figsize=(3.4, 1.9))
    ax.hist(x, bins=40, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.4)
    ax.axvline(base["operating_income"] / 1e3, color=INK, lw=1.1, ls="-")
    ax.axvline(var, color=ORANGE, lw=1.3, ls="--")
    ax.axvline(cvar, color="#9a3a12", lw=1.3, ls=":")
    ymax = ax.get_ylim()[1]
    ax.text(base["operating_income"] / 1e3, ymax * 0.92, " base", fontsize=6.8, va="top")
    ax.text(var, ymax * 0.92, " VaR$_{5\\%}$", fontsize=6.8, va="top")
    ax.text(cvar, ymax * 0.75, "CVaR$_{5\\%}$ ", fontsize=6.8, va="top", ha="right")
    ax.set_xlabel("Operating income, stagflation scenario ($B)")
    ax.set_ylabel("Draws")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_mc.png"), dpi=400)
    print("MC check: mean %.3f VaR %.3f CVaR %.3f ($B), n=%d" % (x.mean(), var, cvar, len(x)))


# --------------------------------------------------------- elasticity -----
def fig_elasticity():
    eh = pd.read_csv(os.path.join(ROOT, "data", "cache", "elasticity_history.csv"), parse_dates=["as_of"])
    pv = eh.pivot(index="as_of", columns="edge", values="elasticity")
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.3), sharex=True)
    for ax, edges in ((axes[0], (("revenue->cogs", "Revenue → COGS", BLUE, "o", "-"),
                                 ("cogs->payables", "COGS → payables", ORANGE, "s", "--"),
                                 ("revenue->receivables", "Revenue → receivables", AQUA, "^", "-."))),
                      (axes[1], (("inflation_yoy->revenue", "CPI → revenue", BLUE, "o", "-"),
                                 ("inflation_yoy->opex", "CPI → opex", ORANGE, "s", "--"),
                                 ("gdp_yoy->revenue", "GDP → revenue", AQUA, "^", "-.")))):
        for col, lab, c, m, ls in edges:
            ax.plot(pv.index, pv[col], color=c, lw=1.3, ls=ls, marker=m, ms=2.8, label=lab)
        ax.grid(axis="y", color=GRID, lw=0.5)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    axes[0].set_ylim(0.28, 1.15)
    axes[0].legend(frameon=False, fontsize=6.2, loc="lower right", ncol=1)
    axes[1].set_ylim(-2.2, 10.5)
    axes[1].set_yticks([0, 3, 6, 9])
    axes[1].legend(frameon=False, fontsize=6.2, loc="upper right", ncol=1)
    axes[0].set_title("Stable edges", fontsize=7.5, loc="left")
    axes[1].set_title("Unstable edges", fontsize=7.5, loc="left")
    axes[1].axhline(0, color=MUTED, lw=0.6)
    fig.tight_layout(pad=0.3, h_pad=0.5)
    fig.savefig(os.path.join(OUT, "fig_elasticity.png"), dpi=400)


# ----------------------------------------------------------- forecast -----
def fig_forecast():
    bt = pd.read_csv(os.path.join(ROOT, "data", "cache", "backtest_revenue.csv"), parse_dates=["period"])
    pv = bt.pivot(index="period", columns="model", values="pred")
    act = bt.groupby("period")["actual"].first()
    fig, ax = plt.subplots(figsize=(3.4, 1.95))
    ax.plot(act.index, act / 1e3, color=INK, lw=1.5, marker="o", ms=2.6, label="Filed")
    ax.plot(pv.index, pv["SeasonalNaive"] / 1e3, color=ORANGE, lw=1.2, ls="--", marker="s", ms=2.4, label="SeasonalNaive")
    ax.plot(pv.index, pv["SeasonalMomentum"] / 1e3, color=BLUE, lw=1.2, ls="-.", marker="^", ms=2.4, label="SeasonalMomentum")
    ax.set_ylabel("Revenue ($B)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", ncol=3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.12)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_forecast.png"), dpi=400)


# ------------------------------------------------------------ tornado -----
def fig_tornado():
    t = pd.read_csv(os.path.join(OUT, "decision_tornado_cash_min_stagflation.csv"))
    t = t.sort_values("swing")
    base = float(pd.read_json(os.path.join(OUT, "summary.json")).loc["stagflation_cash_min", "decision"])
    names = {"buyback_pct_ocf": "Buybacks (±0.25 of OCF)", "dividend_pct_net_income": "Dividends (±0.10 of NI)",
             "capex_pct_revenue": "Capex (±0.02 of revenue)", "demand_shock": "Demand shock (±10%)",
             "effective_tax_rate": "Tax rate (±5 pp)", "fx_shock": "FX shock (±10%)",
             "inflation_yoy": "Inflation (±3 pp)", "gdp_yoy": "GDP growth (±3 pp)",
             "interest_rate_on_debt": "Debt rate (±3 pp)", "supply_cost_shock": "Supply cost (±10%)"}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    y = np.arange(len(t))
    ax.barh(y, (t["low"] - base) / 1e3, left=base / 1e3, color=BLUE, height=0.62, label="Factor lowered")
    ax.barh(y, (t["high"] - base) / 1e3, left=base / 1e3, color=ORANGE, height=0.62, label="Factor raised")
    ax.axvline(base / 1e3, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([names[f] for f in t["factor"]], fontsize=6.4)
    ax.set_xlabel("Lowest cash balance over 8 quarters ($B)")
    ax.grid(axis="x", color=GRID, lw=0.5)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.legend(frameon=False, fontsize=6.4, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig_tornado.png"), dpi=400)
    print("tornado base %.1f, rows %d" % (base, len(t)))


if __name__ == "__main__":
    fig_dag()
    fig_elasticity()
    fig_forecast()
    fig_replay()
    fig_mc()
    fig_tornado()
    print("figures ->", OUT)
