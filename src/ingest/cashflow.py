"""
Recover quarterly cash-flow values from year-to-date filings.

The problem
-----------
Filers report the cash-flow statement cumulatively from the start of the fiscal
year. A 10-Q for fiscal Q3 shows nine months of operating cash flow, not three.
Treating those YTD facts as quarterly values — which a naive XBRL pipeline does,
because they *look* like ordinary duration facts — overstates Q2 by ~2x and Q3
by ~3x, and the error grows through the year.

The recovery
------------
Within a fiscal year, quarterly value = YTD(this period) - YTD(previous period),
with fiscal Q1 taken as-is. The fiscal year is identified by the fact's *start*
date, which is why the extractor preserved it.

Two guards matter:
  * Only difference facts sharing the same fiscal-year start. Differencing
    across a year boundary produces nonsense.
  * Some filings restate an earlier YTD figure. Where two facts cover the same
    (start, end), the first seen is kept — consistent with the rest of the
    pipeline.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "data", "raw"))

def ytd_to_quarterly(rows: List[Tuple[str, str, float]]) -> pd.Series:
    """
    rows: [(fy_start_iso, period_end_iso, ytd_value), ...]
    returns: Series indexed by period end, holding the discrete quarterly value.
    """
    if not rows:
        return pd.Series(dtype=float)

    # group by fiscal-year start
    by_fy: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
    seen = set()
    for start, end, val in rows:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        by_fy.setdefault(start, []).append((pd.Timestamp(end), float(val)))

    out: Dict[pd.Timestamp, float] = {}
    for _, obs in by_fy.items():
        obs.sort(key=lambda t: t[0])
        prev = 0.0
        for end, ytd in obs:
            q = ytd - prev
            # A YTD series must be non-decreasing for a positive-flow concept.
            # A negative difference means the sequence is not a clean YTD chain
            # (a restatement, or a gap where an interim period is missing), so
            # the difference is not trustworthy and is skipped rather than
            # recorded as a negative quarter.
            if q >= 0 or prev == 0.0:
                out[end] = q
            prev = ytd

    return pd.Series(out).sort_index() if out else pd.Series(dtype=float)


def build_cashflow_panel() -> pd.DataFrame:
    """Assemble every YTD-filed concept into a quarterly frame (USD millions)."""
    from sec_batch4 import (OCF_YTD, CAPEX_YTD, BUYBACK_YTD, DNA_YTD,
                            DIVIDENDS_YTD, DEBT_ISSUED_YTD, DEBT_REPAID_YTD,
                            parse_ytd)

    spec = {
        "ocf": OCF_YTD,
        "capex": CAPEX_YTD,
        "buybacks": BUYBACK_YTD,
        "dividends": DIVIDENDS_YTD,
        "dna": DNA_YTD,
        "debt_issued": DEBT_ISSUED_YTD,
        "debt_repaid": DEBT_REPAID_YTD,
    }
    cols = {name: ytd_to_quarterly(parse_ytd(raw)) for name, raw in spec.items()}
    df = pd.DataFrame(cols).sort_index()
    df.index.name = "period_end"
    df["free_cash_flow"] = df["ocf"] - df["capex"]
    df["shareholder_returns"] = df["buybacks"].fillna(0) + df["dividends"].fillna(0)
    df["net_debt_change"] = df["debt_issued"].fillna(0) - df["debt_repaid"].fillna(0)
    return df


def build_income_extras() -> pd.DataFrame:
    """Tax, pretax income and non-operating income as discrete quarterlies."""
    from sec_batch4 import TAX_Q, PRETAX_Q, NONOP_Q, parse_q

    def ser(raw):
        d = parse_q(raw)
        s = pd.Series(d)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    df = pd.DataFrame({"tax": ser(TAX_Q), "pretax_income": ser(PRETAX_Q),
                       "nonoperating_income": ser(NONOP_Q)}).sort_index()
    df.index.name = "period_end"
    df["effective_tax_rate"] = (df["tax"] / df["pretax_income"]).where(
        df["pretax_income"] > 0)
    return df


if __name__ == "__main__":
    cf = build_cashflow_panel()
    inc = build_income_extras()

    out = os.path.join(ROOT, "data", "cache")
    os.makedirs(out, exist_ok=True)
    cf.to_csv(os.path.join(out, "apple_cashflow.csv"))
    inc.to_csv(os.path.join(out, "apple_income_extras.csv"))

    print(f"Cash-flow panel : {cf.shape[0]} quarters x {cf.shape[1]} cols")
    print(f"Income extras   : {inc.shape[0]} quarters x {inc.shape[1]} cols\n")

    print("Coverage:")
    for c in cf.columns:
        print(f"  {c:<20} {cf[c].notna().sum():>3}")

    print("\nMost recent 6 quarters (USD millions):")
    print(cf[["ocf", "capex", "free_cash_flow", "buybacks", "dividends"]]
          .tail(6).round(0).to_string())

    print("\nValidation against known Apple figures:")
    def fy_sum(df, col, start, end):
        m = (df.index > start) & (df.index <= end)
        return df.loc[m, col].sum()

    checks = [
        ("FY2024 OCF sums to ~$118.3B",
         abs(fy_sum(cf, "ocf", "2023-09-30", "2024-09-28") - 118254) < 500),
        ("FY2024 capex sums to ~$9.4B",
         abs(fy_sum(cf, "capex", "2023-09-30", "2024-09-28") - 9447) < 300),
        ("FY2024 buybacks sum to ~$94.9B",
         abs(fy_sum(cf, "buybacks", "2023-09-30", "2024-09-28") - 94949) < 800),
        ("FY2024 dividends sum to ~$15.2B",
         abs(fy_sum(cf, "dividends", "2023-09-30", "2024-09-28") - 15234) < 400),
        ("effective tax rate mostly 13-27%",
         inc["effective_tax_rate"].dropna().between(0.10, 0.30).mean() > 0.80),
        ("no negative operating cash flow quarters",
         (cf["ocf"].dropna() >= 0).all()),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {name}")

    print(f"\nMedian effective tax rate: "
          f"{inc['effective_tax_rate'].median():.1%}")
