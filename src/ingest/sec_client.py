"""
Live SEC EDGAR XBRL ingestion.

This is the production data path: run it on a machine with outbound internet
and it rebuilds the panel directly from primary-source filings, no cached
snapshot involved. It generalises past Apple -- point it at any CIK.

SEC requires a descriptive User-Agent with contact details and rate-limits to
~10 requests/second. Both are honoured below; scraping SEC without a
User-Agent gets the IP blocked.

The three structural quirks this handles are the ones that silently corrupt
naive XBRL pipelines:

  1. Tag migration. Revenue lives under different us-gaap tags across the
     ASC 606 transition, so each field declares a list of candidate tags and
     the results are spliced (verifying that overlapping periods agree).
  2. Missing Q4. Filers never report Q4 as its own duration fact -- it exists
     only inside the 10-K annual figure. Derived as FY - (Q1+Q2+Q3), and only
     when all three predecessors are present.
  3. Duration vs instant. Flow concepts (revenue) carry start+end; stock
     concepts (assets) carry only end. They need different filters, and
     applying the duration filter to a balance-sheet tag silently returns
     nothing.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

SEC_CONCEPT = ("https://data.sec.gov/api/xbrl/companyconcept/"
               "CIK{cik:010d}/us-gaap/{tag}.json")
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# Set a real contact address: SEC blocks anonymous automated traffic.
USER_AGENT = "FinancialDigitalTwin/1.0 (academic project; contact@example.com)"
RATE_LIMIT_S = 0.12  # ~8 req/s, under SEC's ~10 req/s ceiling

# field -> candidate us-gaap tags, in priority order
FIELD_TAGS: Dict[str, List[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax",
                "Revenues", "SalesRevenueNet"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "assets_curr": ["AssetsCurrent"],
    "liabilities": ["Liabilities"],
    "liab_curr": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "payables": ["AccountsPayableCurrent"],
    "lt_debt": ["LongTermDebtNoncurrent"],
}

FLOW_FIELDS = {"revenue", "cogs", "gross_profit", "operating_income", "rnd",
               "net_income"}


class SECClient:
    def __init__(self, user_agent: str = USER_AGENT):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": user_agent,
                               "Accept-Encoding": "gzip, deflate"})
        self._last = 0.0

    def _throttle(self):
        dt = time.time() - self._last
        if dt < RATE_LIMIT_S:
            time.sleep(RATE_LIMIT_S - dt)
        self._last = time.time()

    def cik_for_ticker(self, ticker: str) -> int:
        self._throttle()
        r = self.s.get(SEC_TICKERS, timeout=30)
        r.raise_for_status()
        for row in r.json().values():
            if row["ticker"].upper() == ticker.upper():
                return int(row["cik_str"])
        raise ValueError(f"Ticker not found in SEC index: {ticker}")

    def concept(self, cik: int, tag: str) -> Optional[dict]:
        self._throttle()
        r = self.s.get(SEC_CONCEPT.format(cik=cik, tag=tag), timeout=30)
        if r.status_code == 404:
            return None          # this filer never used this tag
        r.raise_for_status()
        return r.json()


def _split_facts(units: List[dict]):
    """Split USD facts into (quarterly, annual, instant) deduped by end date."""
    def days(a, b):
        return (pd.Timestamp(b) - pd.Timestamp(a)).days

    q, a, inst = {}, {}, {}
    for e in units:
        end = e["end"]
        if e.get("start"):
            d = days(e["start"], end)
            if 80 <= d <= 100:
                q.setdefault(end, e["val"])
            elif 350 <= d <= 380:
                a.setdefault(end, e["val"])
        else:
            inst.setdefault(end, e["val"])

    def ser(d):
        if not d:
            return pd.Series(dtype=float)
        s = pd.Series(d)
        s.index = pd.to_datetime(s.index)
        return s.sort_index().astype(float)

    return ser(q), ser(a), ser(inst)


def _fiscal_quarter(ts: pd.Timestamp, fy_end_month: int = 9):
    """Map a period end to (fiscal_year, quarter) for a given FY-end month."""
    shift = (ts.month - fy_end_month - 1) % 12
    q = shift // 3 + 1
    fy = ts.year + (1 if ts.month > fy_end_month else 0)
    return fy, q


def _derive_q4(q: pd.Series, a: pd.Series, fy_end_month: int = 9) -> pd.Series:
    """Fill Q4 as FY - (Q1+Q2+Q3), only where all three are present."""
    if q.empty or a.empty:
        return q
    have: Dict[int, Dict[int, float]] = {}
    for ts, v in q.items():
        fy, qq = _fiscal_quarter(ts, fy_end_month)
        have.setdefault(fy, {})[qq] = v

    add = {}
    for ts, v in a.items():
        fy, _ = _fiscal_quarter(ts, fy_end_month)
        got = have.get(fy, {})
        if 4 in got or not all(k in got for k in (1, 2, 3)):
            continue
        resid = v - (got[1] + got[2] + got[3])
        if resid > 0:
            add[ts] = resid
    return pd.concat([q, pd.Series(add)]).sort_index() if add else q


def fetch_company(ticker: str = "AAPL", cik: Optional[int] = None,
                  fy_end_month: int = 9, verbose: bool = True) -> pd.DataFrame:
    """Build a quarterly panel (USD millions) for one filer, live from SEC."""
    client = SECClient()
    if cik is None:
        cik = client.cik_for_ticker(ticker)
    if verbose:
        print(f"Fetching {ticker} (CIK {cik:010d}) from SEC EDGAR ...")

    cols = {}
    for field, tags in FIELD_TAGS.items():
        merged_q = pd.Series(dtype=float)
        merged_a = pd.Series(dtype=float)
        merged_i = pd.Series(dtype=float)

        for tag in tags:
            js = client.concept(cik, tag)
            if not js:
                continue
            units = (js.get("units") or {}).get("USD") or []
            if not units:
                continue
            q, a, inst = _split_facts(units)

            # splice, verifying overlapping periods agree before trusting them
            for new, cur, label in ((q, merged_q, "Q"), (a, merged_a, "A"),
                                    (inst, merged_i, "I")):
                if cur.empty:
                    continue
                ov = new.index.intersection(cur.index)
                if len(ov) and not np.allclose(new[ov], cur[ov], rtol=1e-6):
                    print(f"  ! {field}/{tag}: {label} overlap disagrees on "
                          f"{len(ov)} periods; keeping earlier tag")

            merged_q = merged_q.combine_first(q)
            merged_a = merged_a.combine_first(a)
            merged_i = merged_i.combine_first(inst)

        if field in FLOW_FIELDS:
            s = _derive_q4(merged_q, merged_a, fy_end_month)
        else:
            s = merged_i if not merged_i.empty else merged_q

        if not s.empty:
            cols[field] = s / 1e6      # USD -> millions
        if verbose:
            print(f"  {field:<18} {len(s):>3} periods")

    panel = pd.DataFrame(cols).sort_index()
    panel.index.name = "period_end"

    fq = [_fiscal_quarter(ts, fy_end_month) for ts in panel.index]
    panel.insert(0, "fiscal_year", [f[0] for f in fq])
    panel.insert(1, "fiscal_q", [f[1] for f in fq])

    # enforce the gross-profit identity where both sides are reported
    if {"revenue", "gross_profit"} <= set(panel.columns):
        both = panel["revenue"].notna() & panel["gross_profit"].notna()
        panel.loc[both, "cogs"] = (panel.loc[both, "revenue"]
                                   - panel.loc[both, "gross_profit"])
    return panel


if __name__ == "__main__":
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Fetch a filer's quarterly panel from SEC EDGAR")
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--fy-end-month", type=int, default=9,
                    help="Fiscal year end month (Apple=9, most filers=12)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = fetch_company(args.ticker, fy_end_month=args.fy_end_month)
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = args.out or os.path.join(root, "data", "cache",
                                   f"{args.ticker.lower()}_quarterly_panel.csv")
    df.to_csv(out)
    print(f"\n{df.shape[0]} quarters x {df.shape[1]} cols -> {out}")
    print(df.tail(4).to_string())
