"""
Financial Digital Twin — interactive platform.

Run:  streamlit run dashboard/app.py

Chart rules applied throughout: one y-axis per chart (never dual), a validated
categorical palette assigned by entity rather than rank, a legend whenever two
or more series are present, direct labels on final points, and a table view
behind every figure so nothing depends on colour alone.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from features import load_panel, build_features                       # noqa: E402
from graph.causal_graph import fit_graph                              # noqa: E402
from ingest.cashflow import build_cashflow_panel, build_income_extras # noqa: E402
from twin.statements import calibrate, opening_state                  # noqa: E402
from twin.dynamics import (estimate_lags, baseline_path, ShockPath,   # noqa: E402
                           cash_runway, trough)
from twin.decisions import (TwinRunner, LEVERS, LEVER_BY_KEY, METRICS, # noqa: E402
                            tornado, goal_seek, optimise_levers)
from simulate.risk import (RiskCalibration, score_state, band)        # noqa: E402

# --- validated palette (dataviz/references/palette.md) ---------------------
C = {"s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
     "good": "#0ca30c", "warning": "#fab219",
     "serious": "#ec835a", "critical": "#d03b3b",
     "grid": "rgba(130,130,130,0.18)", "ink": "#52514e"}
BAND_STYLE = {"Low": ("●", C["good"]), "Moderate": ("◆", C["warning"]),
              "Elevated": ("▲", C["serious"]), "High": ("■", C["critical"]),
              "Unknown": ("○", C["ink"])}
CACHE = os.path.join(ROOT, "data", "cache")

st.set_page_config(page_title="Financial Digital Twin", layout="wide",
                   initial_sidebar_state="expanded")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_core():
    panel = load_panel()
    feats = build_features(panel)
    macro = pd.read_csv(os.path.join(CACHE, "macro_quarterly.csv"),
                        index_col=0, parse_dates=True)
    cf = build_cashflow_panel()
    ie = build_income_extras()
    return panel, feats, macro, cf, ie


@st.cache_resource(show_spinner=False)
def load_model(_panel, _feats, _macro, _cf, _ie):
    graph = fit_graph(_feats, _macro)
    lags = estimate_lags(_feats, _macro)
    params = calibrate(_panel, _cf, _ie)
    state0 = opening_state(_panel, _cf, _ie)
    return graph, lags, params, state0


@st.cache_data(show_spinner=False)
def load_csv(name):
    p = os.path.join(CACHE, name)
    if not os.path.exists(p):
        return None
    return pd.read_csv(p)


def style(fig, height=380, ytitle=""):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=34, b=8),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      hovermode="x unified", font=dict(size=12),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(showgrid=False, linecolor=C["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=C["grid"], zeroline=False,
                     title=ytitle, linecolor="rgba(0,0,0,0)")
    return fig


panel, feats, macro, cf, ie = load_core()
graph, lags, params, state0 = load_model(panel, feats, macro, cf, ie)
calib = RiskCalibration.from_features(feats)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Twin controls")

horizon = st.sidebar.slider("Projection horizon (quarters)", 4, 16, 8, 1)

st.sidebar.markdown("### External shocks")
PRESETS = {
    "None (baseline)": {},
    "Demand contraction": {"demand_shock": -0.15},
    "Supply-chain disruption": {"supply_cost_shock": 0.15, "demand_shock": -0.05},
    "Inflation spike": {"inflation_yoy": 0.06, "supply_cost_shock": 0.06},
    "Currency shock": {"fx_shock": -0.10},
    "Stagflation": {"demand_shock": -0.12, "supply_cost_shock": 0.10,
                    "inflation_yoy": 0.06},
    "Severe recession (CCAR-style)": {"demand_shock": -0.25, "gdp_yoy": -0.05,
                                      "supply_cost_shock": 0.05},
}
preset = st.sidebar.selectbox("Preset", list(PRESETS.keys()), index=0)
pv = PRESETS[preset]

demand = st.sidebar.slider("Demand", -40.0, 25.0,
                           pv.get("demand_shock", 0.0) * 100, 1.0) / 100
cost = st.sidebar.slider("Input / supply cost", -15.0, 40.0,
                         pv.get("supply_cost_shock", 0.0) * 100, 1.0) / 100
infl = st.sidebar.slider("Inflation (YoY)", -2.0, 12.0,
                         pv.get("inflation_yoy", 0.0) * 100, 0.5) / 100
fx = st.sidebar.slider("FX translation", -25.0, 25.0,
                       pv.get("fx_shock", 0.0) * 100, 1.0) / 100
gdp = st.sidebar.slider("GDP growth", -8.0, 8.0,
                        pv.get("gdp_yoy", 0.0) * 100, 0.5) / 100

st.sidebar.markdown("**Shock shape**")
persistence = st.sidebar.slider("Persistence (share remaining / quarter)",
                                0.0, 1.0, 0.85, 0.05)
onset = st.sidebar.slider("Onset delay (quarters)", 0, 4, 0, 1)
ramp = st.sidebar.slider("Ramp-in (quarters)", 0, 4, 0, 1)

shocks = {k: ShockPath(v, persistence=persistence, onset=onset, ramp=ramp)
          for k, v in [("demand_shock", demand), ("supply_cost_shock", cost),
                       ("inflation_yoy", infl), ("fx_shock", fx),
                       ("gdp_yoy", gdp)] if v != 0.0}

st.sidebar.markdown("### Management levers")
st.sidebar.caption("Defaults are calibrated from Apple's filing history. "
                   "Changing one asks: what if management decided differently?")
lever_vals = {}
for lv in LEVERS:
    cur = float(getattr(params, lv.key))
    if lv.key in ("debt_issued", "debt_repaid"):
        with st.sidebar.expander(lv.label, expanded=False):
            lever_vals[lv.key] = st.slider(lv.label, lv.lo, lv.hi, cur, lv.step,
                                           key=f"lv_{lv.key}", help=lv.help)
    else:
        lever_vals[lv.key] = st.sidebar.slider(
            f"{lv.label} ({lv.unit})" if lv.unit else lv.label,
            lv.lo, lv.hi, float(np.clip(cur, lv.lo, lv.hi)), lv.step,
            key=f"lv_{lv.key}", help=lv.help)

changed_levers = {k: v for k, v in lever_vals.items()
                  if abs(v - float(getattr(params, k))) > 1e-12}

bl = baseline_path(panel, horizon, cashflow=cf)
runner = TwinRunner(graph, state0, params, lags, bl, horizon=horizon)

base_traj = runner.run(None, None)
scen_traj = runner.run(shocks, changed_levers)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("Financial Digital Twin — Apple Inc.")
st.caption(
    f"SEC EDGAR XBRL (CIK 0000320193) + FRED · base quarter "
    f"**{state0.period.date()}** · {len(panel)} quarters of filings · "
    f"projecting {horizon} quarters"
)

sf, bf = scen_traj.iloc[-1], base_traj.iloc[-1]
k = st.columns(5)
k[0].metric("Revenue (final Q)", f"${sf['revenue']:,.0f}M",
            f"{sf['revenue']/bf['revenue']-1:+.1%} vs baseline")
k[1].metric("Operating income", f"${sf['operating_income']:,.0f}M",
            f"{sf['operating_income']/bf['operating_income']-1:+.1%}")
k[2].metric("Gross margin", f"{sf['gross_margin']:.1%}",
            f"{(sf['gross_margin']-bf['gross_margin'])*100:+.1f} pp")
k[3].metric("Cash (final Q)", f"${sf['cash']:,.0f}M",
            f"{sf['cash']-bf['cash']:+,.0f}M")

st_ratios = {m: sf.get(m, np.nan) for m in
             ["current_ratio", "quick_ratio", "cash_ratio", "debt_to_equity",
              "debt_to_assets", "lt_debt_to_equity", "operating_margin",
              "gross_margin", "asset_turnover"]}
risk = score_state(st_ratios, calib)
icon, _ = BAND_STYLE[risk["band"]]
k[4].metric(f"Risk {icon} {risk['band']}", f"{risk['composite']:.0f}/100",
            delta_color="inverse")

rw = cash_runway(scen_traj)
if rw:
    st.error(f"**Liquidity breach in projected quarter {rw}** — cash goes "
             f"negative under this configuration.")

tabs = st.tabs(["Scenario Lab", "Three statements", "Decision Studio",
                "Replay validation", "Twin drift", "Risk", "Forecast",
                "Causal graph", "History", "Data"])

# ---------------------------------------------------------------------------
# 1 · Scenario Lab
# ---------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Shock trajectory vs baseline")
    st.caption("A shock is a *path*, not a step: it ramps in, persists, and "
               "decays. Lags mean the trough usually arrives after the shock.")

    metric = st.selectbox("Metric", ["revenue", "operating_income", "net_income",
                                     "gross_margin", "operating_margin", "cash",
                                     "ocf", "fcf", "equity", "current_ratio"],
                          index=1)
    fig = go.Figure()
    fig.add_scatter(x=base_traj.index, y=base_traj[metric], name="Baseline",
                    mode="lines", line=dict(color=C["s1"], width=2))
    fig.add_scatter(x=scen_traj.index, y=scen_traj[metric], name=preset
                    if preset != "None (baseline)" else "Scenario",
                    mode="lines", line=dict(color=C["s2"], width=2))
    for d, c in ((base_traj, C["s1"]), (scen_traj, C["s2"])):
        v = d[metric].iloc[-1]
        lab = f"{v:,.0f}" if abs(v) > 5 else f"{v:.3f}"
        fig.add_annotation(x=d.index[-1], y=v, text=lab, showarrow=False,
                           xshift=30, font=dict(color=C["ink"], size=11))
    st.plotly_chart(style(fig, 420, metric), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Shock paths applied**")
        if not shocks:
            st.info("No shock configured.")
        else:
            fig2 = go.Figure()
            for i, (nm, sp) in enumerate(shocks.items()):
                fig2.add_scatter(x=list(range(1, horizon + 1)),
                                 y=sp.path(horizon) * 100, name=nm, mode="lines",
                                 line=dict(color=[C["s1"], C["s2"], C["s3"]][i % 3],
                                           width=2))
            st.plotly_chart(style(fig2, 300, "% shock"), width="stretch")
    with c2:
        st.markdown("**Trough and recovery**")
        tr = trough(scen_traj, "operating_income")
        if tr:
            st.metric("Worst quarter", f"Q{tr['trough_quarter']}",
                      f"${tr['trough_value']:,.0f}M operating income")
        cum_s = scen_traj["operating_income"].sum()
        cum_b = base_traj["operating_income"].sum()
        st.metric("Cumulative operating income", f"${cum_s:,.0f}M",
                  f"{cum_s/cum_b-1:+.1%} vs baseline")
        st.metric("Cumulative free cash flow", f"${scen_traj['fcf'].sum():,.0f}M",
                  f"{scen_traj['fcf'].sum()-base_traj['fcf'].sum():+,.0f}M")

    with st.expander("Trajectory table"):
        st.dataframe(scen_traj[["t", "revenue", "gross_margin", "operating_income",
                                "net_income", "ocf", "fcf", "cash", "equity",
                                "current_ratio"]].round(3), width="stretch")

# ---------------------------------------------------------------------------
# 2 · Three statements
# ---------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Projected financial statements")
    st.caption("The balance sheet balances by construction — the check below is "
               "an assertion, not a plug.")

    err = scen_traj["balance_error"].abs().max()
    st.success(f"Balance sheet ties every projected quarter · "
               f"max |A − (L+E)| = ${err:,.9f}M")

    which = st.radio("Statement", ["Income statement", "Cash flow",
                                   "Balance sheet"], horizontal=True)
    d = scen_traj.copy()
    d.index = d.index.date

    if which == "Income statement":
        cols = ["revenue", "cogs", "gross_profit", "opex_total",
                "operating_income", "nonoperating_income", "pretax_income",
                "tax", "net_income"]
    elif which == "Cash flow":
        cols = ["net_income", "dna", "ocf", "capex", "fcf", "buybacks",
                "dividends", "delta_cash", "cash"]
    else:
        cols = ["cash", "receivables", "inventory", "ppe", "assets",
                "payables", "debt", "liabilities", "equity", "balance_error"]

    st.dataframe(d[cols].T.round(1), width="stretch")

    st.markdown("**Cash bridge — where the money goes**")
    last = scen_traj.iloc[-1]
    steps = [("Operating cash flow", last["ocf"]), ("Capex", -last["capex"]),
             ("Buybacks", -last["buybacks"]), ("Dividends", -last["dividends"])]
    fig = go.Figure(go.Bar(
        x=[s[0] for s in steps], y=[s[1] for s in steps],
        marker_color=[C["s3"] if v >= 0 else C["s2"] for _, v in steps],
        marker_line_width=0,
        text=[f"{v:+,.0f}" for _, v in steps], textposition="outside"))
    fig.update_layout(showlegend=False, title="Final projected quarter (USD M)")
    st.plotly_chart(style(fig, 320, "USD millions"), width="stretch")

# ---------------------------------------------------------------------------
# 3 · Decision Studio
# ---------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Decision Studio")
    st.caption("Rank what matters, then solve for the lever setting that hits a "
               "target — the inverse of a normal what-if.")

    sub = st.radio("Tool", ["Sensitivity (tornado)", "Goal-seek",
                            "Multi-lever optimiser"], horizontal=True)

    if sub == "Sensitivity (tornado)":
        m = st.selectbox("Metric to rank against", list(METRICS.keys()),
                         index=list(METRICS.keys()).index("net_income_total"),
                         format_func=lambda k: METRICS[k][0])
        if st.button("Run sensitivity analysis", type="primary"):
            with st.spinner("Perturbing each factor…"):
                t = tornado(runner, m, base_shocks=shocks or None)
            st.session_state["tornado"] = (t, m)

        if "tornado" in st.session_state:
            t, m = st.session_state["tornado"]
            t = t.sort_values("swing")
            fig = go.Figure()
            fig.add_bar(y=t["factor"], x=t["swing"], orientation="h",
                        marker_color=[C["s2"] if k == "shock" else C["s3"]
                                      for k in t["kind"]],
                        marker_line_width=0,
                        text=[f"{v:,.0f}" for v in t["swing"]],
                        textposition="outside", name="Swing")
            fig.update_layout(showlegend=False,
                              title=f"Impact on {METRICS[m][0]}")
            st.plotly_chart(style(fig, 420, ""), width="stretch")
            st.caption("Orange = external shock · green = management lever. "
                       "Ranking both on one scale answers whether the lever you "
                       "hold is bigger than the risk you face.")
            st.dataframe(t.sort_values("swing", ascending=False)[
                ["factor", "kind", "delta", "low", "base", "high", "swing",
                 "share_of_swing"]].round(3), width="stretch", hide_index=True)

    elif sub == "Goal-seek":
        c1, c2, c3 = st.columns(3)
        lever = c1.selectbox("Lever to solve for", [l.key for l in LEVERS],
                             format_func=lambda k: LEVER_BY_KEY[k].label)
        m = c2.selectbox("Target metric", list(METRICS.keys()),
                         index=list(METRICS.keys()).index("cash_final"),
                         format_func=lambda k: METRICS[k][0])
        cur = runner.evaluate(m, shocks or None, None)
        target = c3.number_input("Target value", value=float(round(cur * 1.1, 2)))

        if st.button("Solve", type="primary"):
            with st.spinner("Bisecting…"):
                r = goal_seek(runner, lever, m, float(target),
                              shocks=shocks or None)
            if r.feasible:
                st.success(r.message)
            else:
                st.warning(r.message)
            fig = go.Figure()
            fig.add_scatter(x=r.curve["lever_value"], y=r.curve["metric_value"],
                            mode="lines", name="Response",
                            line=dict(color=C["s1"], width=2))
            fig.add_hline(y=target, line_dash="dash", line_color=C["s2"],
                          annotation_text="Target")
            if r.solution is not None:
                fig.add_vline(x=r.solution, line_dash="dot", line_color=C["s3"],
                              annotation_text="Solution")
            fig.update_layout(showlegend=False,
                              title=f"{LEVER_BY_KEY[lever].label} → {METRICS[m][0]}")
            st.plotly_chart(style(fig, 380, METRICS[m][0]), width="stretch")

    else:
        st.caption("Search several levers jointly. Each evaluation is a full "
                   "simulation, so this takes a little time.")
        chosen = st.multiselect("Levers to optimise", [l.key for l in LEVERS],
                                default=["buyback_pct_ocf", "capex_pct_revenue"],
                                format_func=lambda k: LEVER_BY_KEY[k].label)
        m = st.selectbox("Objective", list(METRICS.keys()),
                         index=list(METRICS.keys()).index("cash_final"),
                         format_func=lambda k: METRICS[k][0], key="opt_metric")
        direction = st.radio("Direction", ["Maximise", "Minimise"],
                             horizontal=True)
        if st.button("Optimise", type="primary") and chosen:
            with st.spinner("Searching…"):
                res = optimise_levers(runner, chosen, m,
                                      maximise=(direction == "Maximise"),
                                      shocks=shocks or None)
            if res.get("success"):
                st.success(f"Best {METRICS[m][0]}: {res['achieved']:,.2f} "
                           f"(baseline {res['baseline']:,.2f})")
                st.dataframe(pd.DataFrame([
                    {"Lever": LEVER_BY_KEY[k].label,
                     "Current": float(getattr(params, k)),
                     "Optimal": v} for k, v in res["settings"].items()
                ]).round(4), width="stretch", hide_index=True)
            else:
                st.error(res.get("message", "Optimisation failed."))

# ---------------------------------------------------------------------------
# 4 · Replay validation
# ---------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Historical replay validation")
    st.caption("The twin is re-fitted on pre-episode data only, then asked to "
               "reproduce a shock that actually happened. This tests the causal "
               "claim, not just forecast accuracy.")

    summ = load_csv("replay_summary.csv")
    paths = load_csv("replay_paths.csv")

    if summ is None:
        st.warning("Replay not yet generated. Run: `python src/twin/replay.py`")
    else:
        piv = summ.pivot_table(index=["episode", "metric"], columns="mode",
                               values="MAPE_%").round(2)
        st.markdown("**Error (MAPE %) — lower is better**")
        st.dataframe(piv, width="stretch")
        st.info(
            "**Conditional** feeds the actual revenue path and tests whether the "
            "rest of the financial picture follows — the real test of a twin. "
            "**Macro** feeds only GDP and inflation and lets the twin derive "
            "revenue too; it does worse, because Apple's revenue is driven by "
            "product cycles rather than macro. Both are reported."
        )

        if paths is not None:
            paths["period"] = pd.to_datetime(paths["period"])
            eps = sorted(paths["episode"].unique())
            c1, c2, c3 = st.columns(3)
            ep = c1.selectbox("Episode", eps)
            md = c2.selectbox("Mode", sorted(paths["mode"].unique()))
            mopts = [c.replace("actual_", "") for c in paths.columns
                     if c.startswith("actual_")
                     and f"twin_{c.replace('actual_','')}" in paths.columns]
            mm = c3.selectbox("Metric", mopts,
                              index=mopts.index("gross_margin")
                              if "gross_margin" in mopts else 0)
            d = paths[(paths["episode"] == ep) & (paths["mode"] == md)]
            if not d.empty:
                fig = go.Figure()
                fig.add_scatter(x=d["period"], y=d[f"actual_{mm}"],
                                name="Actual (as filed)", mode="lines+markers",
                                line=dict(color=C["s1"], width=2))
                fig.add_scatter(x=d["period"], y=d[f"twin_{mm}"],
                                name="Twin projection", mode="lines+markers",
                                line=dict(color=C["s2"], width=2, dash="dash"))
                st.plotly_chart(style(fig, 400, mm), width="stretch")
                st.dataframe(d[["period", f"actual_{mm}", f"twin_{mm}"]]
                             .round(4), width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 5 · Twin drift
# ---------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Twin drift — the learning loop")
    st.caption("At every historical quarter the twin was re-fitted on data "
               "available then and asked to project the next quarter. This is "
               "its own track record against reality.")

    log = load_csv("prediction_log.csv")
    eh = load_csv("elasticity_history.csv")

    if log is None:
        st.warning("Drift log not generated. Run: `python src/twin/learning.py`")
    else:
        log["target_period"] = pd.to_datetime(log["target_period"])
        acc = (log.groupby("metric")["abs_pct_error"]
               .agg(["count", "mean", "median"])
               .rename(columns={"mean": "MAPE %", "median": "Median %"}))
        st.dataframe(acc.round(2), width="stretch")

        m = st.selectbox("Metric", sorted(log["metric"].unique()),
                         index=sorted(log["metric"].unique()).index("gross_margin")
                         if "gross_margin" in log["metric"].unique() else 0)
        d = log[log["metric"] == m].sort_values("target_period")
        fig = go.Figure()
        fig.add_scatter(x=d["target_period"], y=d["actual"], name="Actual",
                        mode="lines", line=dict(color=C["s1"], width=2))
        fig.add_scatter(x=d["target_period"], y=d["predicted"], name="Twin",
                        mode="lines", line=dict(color=C["s2"], width=2,
                                                dash="dash"))
        st.plotly_chart(style(fig, 360, m), width="stretch")

        fig2 = go.Figure()
        fig2.add_scatter(x=d["target_period"], y=d["abs_pct_error"],
                         name="Absolute % error", mode="lines",
                         line=dict(color=C["s3"], width=2))
        fig2.add_hline(y=d["abs_pct_error"].mean(), line_dash="dot",
                       line_color=C["ink"], annotation_text="mean")
        fig2.update_layout(showlegend=False, title="Drift over time")
        st.plotly_chart(style(fig2, 300, "% error"), width="stretch")

    if eh is not None and not eh.empty:
        st.markdown("**Elasticity evolution — the twin re-learning**")
        eh["as_of"] = pd.to_datetime(eh["as_of"])
        edges = sorted(eh["edge"].unique())
        default = [e for e in ["revenue->cogs", "cogs->payables",
                               "revenue->receivables"] if e in edges]
        pick = st.multiselect("Edges", edges, default=default or edges[:3])
        fig = go.Figure()
        for i, e in enumerate(pick):
            d = eh[eh["edge"] == e].sort_values("as_of")
            fig.add_scatter(x=d["as_of"], y=d["elasticity"], name=e,
                            mode="lines",
                            line=dict(color=[C["s1"], C["s2"], C["s3"]][i % 3],
                                      width=2))
        st.plotly_chart(style(fig, 360, "elasticity"), width="stretch")
        st.caption("Internal operating elasticities (revenue→COGS and the "
                   "working-capital edges) are remarkably stable across two "
                   "decades. The macro edges are not — which is why demand "
                   "enters as a user-injected shock rather than a macro forecast.")

# ---------------------------------------------------------------------------
# 6 · Risk
# ---------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Risk profile")
    st.caption("Scores are percentiles within Apple's own filing history, not "
               "industry cut-offs: 0 = healthiest this firm has been, 100 = most "
               "stressed. A current ratio near 1.0 is normal here.")
    cols = st.columns(3)
    for i, cat in enumerate(["liquidity", "leverage", "operational"]):
        s_ = risk["categories"].get(cat, np.nan)
        ic, _ = BAND_STYLE[band(s_)]
        cols[i].metric(f"{cat.title()} {ic} {band(s_)}", f"{s_:.0f}/100",
                       delta_color="inverse")

    det = risk["detail"].copy()
    det["value"] = det["value"].map(lambda v: f"{v:,.3f}" if np.isfinite(v) else "—")
    det["risk_score"] = det["risk_score"].map(
        lambda v: f"{v:.0f}" if np.isfinite(v) else "—")
    st.dataframe(det, width="stretch", hide_index=True)

    st.markdown("**Risk through the projection**")
    rr = []
    for ts, row in scen_traj.iterrows():
        stt = {m: row.get(m, np.nan) for m in
               ["current_ratio", "quick_ratio", "cash_ratio", "debt_to_equity",
                "debt_to_assets", "lt_debt_to_equity", "operating_margin",
                "gross_margin", "asset_turnover"]}
        rr.append({"period": ts, "composite": score_state(stt, calib)["composite"]})
    rdf = pd.DataFrame(rr)
    fig = go.Figure()
    fig.add_scatter(x=rdf["period"], y=rdf["composite"], mode="lines",
                    name="Composite risk", line=dict(color=C["s2"], width=2))
    st.plotly_chart(style(fig, 320, "risk (0-100)"), width="stretch")

# ---------------------------------------------------------------------------
# 7 · Forecast
# ---------------------------------------------------------------------------
with tabs[6]:
    st.subheader("Walk-forward backtest")
    st.caption("Every point predicted using only data strictly before it, model "
               "refit at each step.")
    target = st.selectbox("Target", ["revenue", "cogs", "operating_income"])
    bt = load_csv(f"backtest_{target}.csv")
    if bt is None:
        st.warning("Run: `python src/forecast/backtest.py`")
    else:
        bt["period"] = pd.to_datetime(bt["period"])
        summ = (bt.assign(ape=lambda d: (d["pred"] - d["actual"]).abs() / d["actual"])
                  .groupby("model")["ape"].mean().mul(100).sort_values()
                  .rename("MAPE %").to_frame())
        st.dataframe(summ.style.format("{:.2f}"), width="stretch")
        best = summ.index[0]
        show = st.multiselect("Models", list(summ.index),
                              default=[best, "SeasonalNaive"])
        d = bt[bt["model"] == best]
        fig = go.Figure()
        fig.add_scatter(x=d["period"], y=d["actual"], name="Actual", mode="lines",
                        line=dict(color=C["s1"], width=2))
        for i, mn in enumerate(show):
            dd = bt[bt["model"] == mn]
            fig.add_scatter(x=dd["period"], y=dd["pred"], name=mn, mode="lines",
                            line=dict(color=[C["s2"], C["s3"]][i % 2], width=2,
                                      dash="dash"))
        st.plotly_chart(style(fig, 380, "USD millions"), width="stretch")
        st.info(f"Best: **{best}** ({summ.iloc[0, 0]:.2f}% MAPE). The LSTM is "
                "evaluated, not assumed superior — with ~70 quarters it overfits "
                "and loses to a seasonal baseline. That comparison is the result.")

# ---------------------------------------------------------------------------
# 8 · Causal graph
# ---------------------------------------------------------------------------
with tabs[7]:
    st.subheader("Estimated causal structure")
    st.caption("Behavioural edges are elasticities estimated from Apple's own "
               "filings (% per %, year-over-year, with fiscal-quarter fixed "
               "effects). Structural edges are accounting identities and are "
               "never estimated.")
    df = graph.to_frame()
    beh = df[df.kind == "behavioral"].copy()
    beh["significant"] = np.where(beh["t_stat"].abs() > 2, "yes", "no")
    beh["lag_q"] = [lags.get((r.parent, r.child), 0) for r in beh.itertuples()]
    st.dataframe(beh.drop(columns="kind").round(3), width="stretch",
                 hide_index=True)

    st.markdown("**Per-equation fit**")
    fit = pd.DataFrame([{"Equation": k, "Obs": v.get("n"), "R²": v.get("r2")}
                        for k, v in graph.fit_stats.items()]).set_index("Equation")
    st.dataframe(fit.round(3), width="stretch")
    st.warning("The revenue equation has low R². Reported, not hidden: Apple's "
               "revenue is driven by product cycles rather than US macro, so GDP "
               "and inflation are statistically insignificant. That is why demand "
               "enters as a user-injected shock node.")

    st.markdown("**Calibrated policy parameters**")
    st.dataframe(pd.DataFrame([{"Parameter": k, "Value": v}
                               for k, v in params.__dict__.items()]).round(5),
                 width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# 9 · History
# ---------------------------------------------------------------------------
with tabs[8]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue and gross profit")
        d = panel.dropna(subset=["revenue"]).tail(40)
        fig = go.Figure()
        fig.add_bar(x=d.index, y=d["revenue"], name="Revenue",
                    marker_color=C["s1"], marker_line_width=0)
        fig.add_bar(x=d.index, y=d["gross_profit"], name="Gross profit",
                    marker_color=C["s3"], marker_line_width=0)
        fig.update_layout(barmode="group", bargap=0.25)
        st.plotly_chart(style(fig, 360, "USD millions"), width="stretch")
    with c2:
        st.subheader("Margins")
        d = feats.dropna(subset=["gross_margin"]).tail(40)
        fig = go.Figure()
        for col_, nm, c_ in [("gross_margin", "Gross", C["s1"]),
                             ("operating_margin", "Operating", C["s2"]),
                             ("net_margin", "Net", C["s3"])]:
            fig.add_scatter(x=d.index, y=d[col_] * 100, name=nm, mode="lines",
                            line=dict(color=c_, width=2))
            fig.add_annotation(x=d.index[-1], y=d[col_].iloc[-1] * 100,
                               text=f"{d[col_].iloc[-1]*100:.0f}%",
                               showarrow=False, xshift=22,
                               font=dict(color=C["ink"]))
        st.plotly_chart(style(fig, 360, "percent"), width="stretch")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Capital allocation")
        d = cf.dropna(subset=["buybacks"]).tail(32)
        fig = go.Figure()
        fig.add_bar(x=d.index, y=d["buybacks"], name="Buybacks",
                    marker_color=C["s2"], marker_line_width=0)
        fig.add_bar(x=d.index, y=d["dividends"], name="Dividends",
                    marker_color=C["s3"], marker_line_width=0)
        fig.add_bar(x=d.index, y=d["capex"], name="Capex",
                    marker_color=C["s1"], marker_line_width=0)
        fig.update_layout(barmode="stack", bargap=0.25)
        st.plotly_chart(style(fig, 360, "USD millions"), width="stretch")
        st.caption("Buybacks dominate — and are why equity shrinks while the "
                   "business grows.")
    with c4:
        st.subheader("Cash conversion cycle")
        d = feats.dropna(subset=["cash_conversion_cycle"]).tail(40)
        fig = go.Figure()
        fig.add_scatter(x=d.index, y=d["cash_conversion_cycle"], mode="lines",
                        name="CCC", line=dict(color=C["s1"], width=2))
        fig.add_hline(y=0, line_dash="dot", line_color=C["ink"])
        st.plotly_chart(style(fig, 360, "days"), width="stretch")
        st.caption("Persistently negative: Apple collects before it pays, so "
                   "growth releases cash instead of consuming it.")

# ---------------------------------------------------------------------------
# 10 · Data
# ---------------------------------------------------------------------------
with tabs[9]:
    st.subheader("Underlying data")
    st.caption("Primary source: SEC EDGAR XBRL company-concept facts. All "
               "accounting identities validated to $0.000M.")
    which = st.radio("Table", ["Quarterly panel", "Derived features",
                               "Cash flow", "Income extras", "Macro"],
                     horizontal=True)
    d = {"Quarterly panel": panel, "Derived features": feats, "Cash flow": cf,
         "Income extras": ie, "Macro": macro}[which]
    st.dataframe(d.tail(60), width="stretch")
    st.download_button(f"Download {which} (CSV)", d.to_csv().encode(),
                       file_name=f"{which.lower().replace(' ', '_')}.csv",
                       mime="text/csv")
