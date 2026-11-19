"""
Derive the Financial Digital Twin's state variables from the raw SEC panel.

The panel gives raw line items. The *twin* reasons over derived state: margins,
liquidity, efficiency, leverage and growth. These are the nodes the causal graph
propagates shocks through, and the targets the forecasting layer predicts.

All monetary inputs are USD millions.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(ROOT, "data", "cache", "apple_quarterly_panel.csv")

# Quarter length in days, used to annualise flow-vs-stock ratios (DSO/DIO/DPO).
DAYS_PER_QUARTER = 91.25


def load_panel(path: str = PANEL_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "period_end"
    return df.sort_index()


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Divide, returning NaN where the denominator is zero/NaN rather than inf."""
    b = b.replace(0, np.nan)
    return a / b


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel
    f = pd.DataFrame(index=p.index)
    f["fiscal_year"] = p["fiscal_year"]
    f["fiscal_q"] = p["fiscal_q"]

    # ---- scale ------------------------------------------------------------
    f["revenue"] = p["revenue"]
    f["net_income"] = p["net_income"]
    f["operating_income"] = p["operating_income"]

    # ---- profitability ----------------------------------------------------
    f["gross_margin"] = _safe_div(p["gross_profit"], p["revenue"])
    f["operating_margin"] = _safe_div(p["operating_income"], p["revenue"])
    f["net_margin"] = _safe_div(p["net_income"], p["revenue"])
    f["rnd_intensity"] = _safe_div(p["rnd"], p["revenue"])

    # opex implied by the income statement (gross profit less operating income)
    f["opex"] = p["gross_profit"] - p["operating_income"]
    f["opex_ratio"] = _safe_div(f["opex"], p["revenue"])

    # ---- liquidity --------------------------------------------------------
    f["current_ratio"] = _safe_div(p["assets_curr"], p["liab_curr"])
    f["quick_ratio"] = _safe_div(p["assets_curr"] - p["inventory"], p["liab_curr"])
    f["cash_ratio"] = _safe_div(p["cash"], p["liab_curr"])
    f["working_capital"] = p["assets_curr"] - p["liab_curr"]

    # ---- leverage / solvency ---------------------------------------------
    f["debt_to_equity"] = _safe_div(p["liabilities"], p["equity"])
    f["debt_to_assets"] = _safe_div(p["liabilities"], p["assets"])
    f["equity_ratio"] = _safe_div(p["equity"], p["assets"])
    f["lt_debt_to_equity"] = _safe_div(p["lt_debt"], p["equity"])

    # ---- efficiency: cash conversion cycle --------------------------------
    # DSO/DIO/DPO use the quarter's flow annualised to that quarter's length.
    f["dso"] = _safe_div(p["receivables"], p["revenue"]) * DAYS_PER_QUARTER
    f["dio"] = _safe_div(p["inventory"], p["cogs"]) * DAYS_PER_QUARTER
    f["dpo"] = _safe_div(p["payables"], p["cogs"]) * DAYS_PER_QUARTER
    f["cash_conversion_cycle"] = f["dso"] + f["dio"] - f["dpo"]

    f["asset_turnover"] = _safe_div(p["revenue"], p["assets"])
    f["inventory_turnover"] = _safe_div(p["cogs"], p["inventory"])

    # ---- returns ----------------------------------------------------------
    f["roa"] = _safe_div(p["net_income"], p["assets"])
    f["roe"] = _safe_div(p["net_income"], p["equity"])

    # ---- growth -----------------------------------------------------------
    # YoY on a 4-quarter lag is the meaningful comparison for a business with
    # this much fiscal seasonality (Q1 holiday quarter dominates).
    f["revenue_yoy"] = p["revenue"].pct_change(4)
    f["revenue_qoq"] = p["revenue"].pct_change(1)
    f["net_income_yoy"] = p["net_income"].pct_change(4)
    f["revenue_ttm"] = p["revenue"].rolling(4).sum()
    f["net_income_ttm"] = p["net_income"].rolling(4).sum()

    # ---- log-differences: the regression basis for causal elasticities -----
    # dlog_  = quarter-over-quarter. Heavily seasonal for this business: the
    #          fiscal-Q1 holiday quarter runs ~50% above Q3, which swamps any
    #          macro signal.
    # dlog4_ = year-over-year (4-quarter). Seasonality differences out, and the
    #          horizon matches the macro drivers (which are already YoY rates),
    #          so this is the basis the causal graph estimates on.
    lvl_cols = ["revenue", "cogs", "operating_income", "inventory",
                "receivables", "payables", "assets", "equity"]
    for col in lvl_cols:
        s = p[col]
        ls = np.log(s.where(s > 0))
        f[f"dlog_{col}"] = ls.diff()
        f[f"dlog4_{col}"] = ls.diff(4)

    # opex is derived, not a raw panel column -- give it the same treatment
    lo = np.log(f["opex"].where(f["opex"] > 0))
    f["dlog_opex"] = lo.diff()
    f["dlog4_opex"] = lo.diff(4)

    return f


def summarise(f: pd.DataFrame) -> pd.DataFrame:
    """Coverage + latest value for each derived feature (for sanity checks)."""
    rows = []
    for c in f.columns:
        if c in ("fiscal_year", "fiscal_q"):
            continue
        s = f[c].dropna()
        rows.append({
            "feature": c,
            "n": len(s),
            "latest": s.iloc[-1] if len(s) else np.nan,
            "mean": s.mean() if len(s) else np.nan,
            "std": s.std() if len(s) else np.nan,
        })
    return pd.DataFrame(rows).set_index("feature")


if __name__ == "__main__":
    panel = load_panel()
    f = build_features(panel)

    out = os.path.join(ROOT, "data", "cache", "apple_features.csv")
    f.to_csv(out)
    print(f"Features: {f.shape[0]} quarters x {f.shape[1]} columns -> {out}\n")

    key = ["gross_margin", "operating_margin", "net_margin", "current_ratio",
           "quick_ratio", "debt_to_equity", "cash_conversion_cycle", "roe",
           "revenue_yoy"]
    print("Latest 4 quarters (key ratios):")
    print(f[["fiscal_year", "fiscal_q"] + key].tail(4).round(3).to_string())

    print("\nSanity checks against known Apple characteristics:")
    latest = f.dropna(subset=["gross_margin"]).iloc[-1]
    checks = [
        ("gross margin in 40-50% band", 0.40 <= latest["gross_margin"] <= 0.52),
        ("operating margin in 25-40% band", 0.25 <= latest["operating_margin"] <= 0.40),
        ("current ratio in 0.7-1.5 band", 0.7 <= latest["current_ratio"] <= 1.5),
        ("negative cash conversion cycle", latest["cash_conversion_cycle"] < 0),
        ("ROE elevated (buybacks shrink equity)", latest["roe"] > 0.10),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {name}")
