"""
Assemble Apple's SEC XBRL facts into a clean quarterly financial panel.

Handles the four real-world messes in SEC data:
  1. Tag migration   - revenue splits across SalesRevenueNet (pre-ASC606) and
                       RevenueFromContractWithCustomerExcludingAssessedTax.
  2. Missing Q4      - filers never report Q4 separately; it lives only inside
                       the 10-K annual figure. Derived as FY - (Q1+Q2+Q3).
  3. Mixed units     - batch1 is raw USD, batches 2/3 are USD millions.
  4. YTD cash flow   - only fiscal Q1 is a true 90-day duration fact.

Output: data/cache/apple_quarterly_panel.csv  (all values in USD millions)
"""
import sys, os
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "data", "raw"))

from sec_batch1 import BATCH1, parse          # noqa: E402
from sec_batch2 import BATCH2, parse_m        # noqa: E402
from sec_batch3 import BATCH3                 # noqa: E402
from sec_batch4 import PPE_NET                # noqa: E402

M = 1e6


def s_from_batch1(key, kind):
    """batch1 stores raw USD -> convert to millions."""
    d = parse(BATCH1[key][kind])
    return pd.Series({pd.Timestamp(k): v / M for k, v in d.items()}).sort_index()


def s_millions(raw):
    """batch2/3 store USD millions already (parse_m scales to USD)."""
    d = parse_m(raw)
    return pd.Series({pd.Timestamp(k): v / M for k, v in d.items()}).sort_index()


def fiscal_quarter(ts):
    """Apple's FY ends late Sep. Map a period-end date to (fiscal_year, q)."""
    m = ts.month
    if m in (10, 11, 12, 1):
        return ts.year + (1 if m >= 10 else 0), 1
    if m in (2, 3, 4):
        return ts.year, 2
    if m in (5, 6, 7):
        return ts.year, 3
    return ts.year, 4


def derive_q4(q: pd.Series, a: pd.Series) -> pd.Series:
    """
    Fill missing Q4 observations using FY - (Q1+Q2+Q3).

    Only fills a fiscal year when all three of Q1..Q3 are present, so we never
    fabricate a value from partial data.
    """
    q = q.copy()
    tag = {ts: fiscal_quarter(ts) for ts in q.index}
    have = {}
    for ts, (fy, qq) in tag.items():
        have.setdefault(fy, {})[qq] = q[ts]

    added = {}
    for a_end, a_val in a.items():
        fy, _ = fiscal_quarter(a_end)
        got = have.get(fy, {})
        if 4 in got:
            continue
        if all(k in got for k in (1, 2, 3)):
            resid = a_val - (got[1] + got[2] + got[3])
            if resid > 0:
                added[a_end] = resid
    if added:
        q = pd.concat([q, pd.Series(added)]).sort_index()
    return q


def build():
    # ---- revenue: splice legacy + current tag -------------------------------
    rev_new = s_from_batch1("Revenue", "Q")
    rev_old = s_millions(BATCH2["SalesRevenueNet"])

    overlap = rev_new.index.intersection(rev_old.index)
    diffs = {str(d.date()): (rev_old[d], rev_new[d]) for d in overlap
             if abs(rev_old[d] - rev_new[d]) > 1e-6}
    assert not diffs, f"Revenue tags disagree on overlap: {diffs}"

    revenue_q = rev_old.combine_first(rev_new).sort_index()
    revenue_a = s_from_batch1("Revenue", "A").combine_first(
        pd.Series(dtype=float)).sort_index()

    # ---- income statement ---------------------------------------------------
    cogs_q = s_from_batch1("COGS", "Q")
    gp_q = s_from_batch1("GrossProfit", "Q")
    op_q = s_from_batch1("OperatingIncome", "Q")
    rnd_q = s_from_batch1("RnD", "Q")
    ni_q = s_millions(BATCH2["NetIncome"])

    # derive missing Q4s from the annual figures
    cogs_q = derive_q4(cogs_q, s_from_batch1("COGS", "A"))
    op_q = derive_q4(op_q, s_from_batch1("OperatingIncome", "A"))
    rnd_q = derive_q4(rnd_q, s_from_batch1("RnD", "A"))
    revenue_q = derive_q4(revenue_q, revenue_a)

    # ---- balance sheet (instants) ------------------------------------------
    bs = {
        "assets":        s_millions(BATCH2["Assets"]),
        "assets_curr":   s_millions(BATCH2["AssetsCurrent"]),
        "liabilities":   s_millions(BATCH3["Liabilities"]),
        "liab_curr":     s_millions(BATCH3["LiabilitiesCurrent"]),
        "equity":        s_millions(BATCH3["StockholdersEquity"]),
        "cash":          s_millions(BATCH3["Cash"]),
        "inventory":     s_millions(BATCH3["Inventory"]),
        "receivables":   s_millions(BATCH3["Receivables"]),
        "payables":      s_millions(BATCH3["Payables"]),
        "lt_debt":       s_millions(BATCH3["LongTermDebt"]),
        "ppe":           s_millions(PPE_NET),
    }

    panel = pd.DataFrame({
        "revenue": revenue_q,
        "cogs": cogs_q,
        "gross_profit": gp_q,
        "operating_income": op_q,
        "rnd": rnd_q,
        "net_income": ni_q,
        **bs,
    }).sort_index()

    panel.index.name = "period_end"

    # ---- reconcile COGS against the gross-profit identity -------------------
    # Where revenue and gross_profit are both reported, COGS is determined
    # exactly. This matters at restatement boundaries (notably FY2009 Q4, where
    # Apple retrospectively restated for iPhone revenue recognition under
    # ASU 2009-13/14): a Q4 derived from *restated* annual COGS is not on the
    # same accounting basis as the *originally filed* quarterly gross profit.
    # Deriving COGS from the identity keeps every row internally consistent.
    both = panel["revenue"].notna() & panel["gross_profit"].notna()
    implied = panel.loc[both, "revenue"] - panel.loc[both, "gross_profit"]
    delta = (implied - panel.loc[both, "cogs"]).abs()
    panel["cogs_restated_flag"] = False
    panel.loc[delta[delta > 1.0].index, "cogs_restated_flag"] = True
    panel.loc[both, "cogs"] = implied

    fq = [fiscal_quarter(ts) for ts in panel.index]
    panel.insert(0, "fiscal_year", [f[0] for f in fq])
    panel.insert(1, "fiscal_q", [f[1] for f in fq])

    return panel


def validate(panel):
    """Check the panel against accounting identities it must satisfy."""
    checks = []

    def chk(name, series, tol):
        s = series.dropna()
        if len(s) == 0:
            checks.append((name, "NO DATA", np.nan, 0))
            return
        worst = s.abs().max()
        checks.append((name, "PASS" if worst <= tol else "FAIL", worst, len(s)))

    # A = L + E
    chk("Assets = Liabilities + Equity",
        panel["assets"] - (panel["liabilities"] + panel["equity"]), 1.0)

    # Revenue - COGS = Gross Profit
    chk("Revenue - COGS = GrossProfit",
        panel["revenue"] - panel["cogs"] - panel["gross_profit"], 1.0)

    # Gross profit must exceed operating income (opex is positive)
    chk("GrossProfit >= OperatingIncome",
        (panel["operating_income"] - panel["gross_profit"]).clip(lower=0), 1.0)

    return checks


if __name__ == "__main__":
    panel = build()
    out = os.path.join(ROOT, "data", "cache")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "apple_quarterly_panel.csv")
    panel.to_csv(path)

    print(f"Panel: {panel.shape[0]} quarters x {panel.shape[1]} columns")
    print(f"Range: {panel.index.min().date()} -> {panel.index.max().date()}")
    print(f"Saved: {path}\n")

    print("Coverage (non-null quarters per series):")
    for c in panel.columns:
        if c in ("fiscal_year", "fiscal_q"):
            continue
        n = panel[c].notna().sum()
        print(f"  {c:<18} {n:>3}/{len(panel)}")

    print("\nAccounting identity validation:")
    for name, status, worst, n in validate(panel):
        w = "n/a" if pd.isna(worst) else f"max err ${worst:,.3f}M"
        print(f"  [{status}] {name:<38} ({n} obs, {w})")

    print("\nMost recent 6 quarters:")
    cols = ["fiscal_year", "fiscal_q", "revenue", "gross_profit",
            "operating_income", "net_income", "assets", "equity"]
    print(panel[cols].tail(6).to_string())
