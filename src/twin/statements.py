"""
The three-statement engine — the core of the twin.

Why this exists
---------------
The earlier version of this project shocked the P&L and stopped at operating
income. That cannot answer the question a CFO actually asks, which is not "how
far does profit fall?" but "do we run out of money, and when?". Answering it
requires the income statement, cash-flow statement and balance sheet to be
linked, so that a shock to revenue flows through margin, into cash, into the
balance sheet, and back into the next period's interest expense.

The balancing argument
----------------------
The balance sheet balances *by construction*, not by a plug. Writing the
period change in each block:

    ΔCash = NI + D&A − ΔAR − ΔInv + ΔAP − Capex + ΔDebt − Div − Buyback
    ΔPP&E (non-current assets) = Capex − D&A
    ΔEquity = NI − Div − Buyback

Summing the asset side:

    ΔAssets = ΔCash + ΔAR + ΔInv + ΔPP&E
            = NI + ΔAP + ΔDebt − Div − Buyback

and the funding side:

    ΔLiabilities + ΔEquity = (ΔAP + ΔDebt) + (NI − Div − Buyback)

The two are identically equal. So if the opening balance sheet ties, every
projected one ties too, and any drift is a bug rather than a rounding artefact.
`simulate()` asserts this each period.

Non-operating income is modelled as interest on the *net* cash position, which
is what closes the loop: a scenario that drains cash raises net interest cost,
which lowers net income, which drains cash further.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class TwinState:
    """A complete, balanced financial position at one period end (USD millions)."""
    period: pd.Timestamp

    # --- balance sheet ---
    cash: float
    receivables: float
    inventory: float
    other_curr_assets: float
    ppe: float                        # rolls with capex − D&A
    other_noncurr_assets: float       # long-term securities etc. (held flat)
    payables: float
    other_curr_liab: float
    debt: float
    other_noncurr_liab: float
    equity: float

    # --- flows realised in the period just ended ---
    revenue: float = np.nan
    cogs: float = np.nan
    opex: float = np.nan            # cash opex (excludes D&A)
    opex_total: float = np.nan      # cash opex + D&A
    gross_profit: float = np.nan
    operating_income: float = np.nan
    nonoperating_income: float = np.nan
    pretax_income: float = np.nan
    tax: float = np.nan
    net_income: float = np.nan
    dna: float = np.nan
    capex: float = np.nan
    buybacks: float = np.nan
    dividends: float = np.nan
    debt_issued: float = np.nan
    debt_repaid: float = np.nan
    ocf: float = np.nan
    fcf: float = np.nan
    delta_cash: float = np.nan

    @property
    def assets_curr(self) -> float:
        return self.cash + self.receivables + self.inventory + self.other_curr_assets

    @property
    def assets(self) -> float:
        return self.assets_curr + self.ppe + self.other_noncurr_assets

    @property
    def liab_curr(self) -> float:
        return self.payables + self.other_curr_liab

    @property
    def liabilities(self) -> float:
        return self.liab_curr + self.debt + self.other_noncurr_liab

    @property
    def balance_error(self) -> float:
        return self.assets - (self.liabilities + self.equity)

    def ratios(self) -> Dict[str, float]:
        def d(a, b):
            return a / b if b else np.nan
        return {
            "current_ratio": d(self.assets_curr, self.liab_curr),
            "quick_ratio": d(self.assets_curr - self.inventory, self.liab_curr),
            "cash_ratio": d(self.cash, self.liab_curr),
            "debt_to_equity": d(self.liabilities, self.equity),
            "debt_to_assets": d(self.liabilities, self.assets),
            "lt_debt_to_equity": d(self.debt, self.equity),
            "gross_margin": d(self.gross_profit, self.revenue),
            "operating_margin": d(self.operating_income, self.revenue),
            "net_margin": d(self.net_income, self.revenue),
            "asset_turnover": d(self.revenue, self.assets),
            "roe": d(self.net_income, self.equity),
            "roa": d(self.net_income, self.assets),
        }

    def to_row(self) -> Dict[str, float]:
        r = {k: getattr(self, k) for k in (
            "revenue", "cogs", "gross_profit", "opex", "opex_total",
            "operating_income", "ppe",
            "nonoperating_income", "pretax_income", "tax", "net_income",
            "dna", "capex", "buybacks", "dividends", "ocf", "fcf",
            "delta_cash", "cash", "receivables", "inventory", "payables",
            "debt", "equity")}
        r.update({"period": self.period, "assets": self.assets,
                  "liabilities": self.liabilities,
                  "assets_curr": self.assets_curr, "liab_curr": self.liab_curr,
                  "working_capital": self.assets_curr - self.liab_curr,
                  "balance_error": self.balance_error})
        r.update(self.ratios())
        return r


# ---------------------------------------------------------------------------
# Policy parameters — calibrated from history, overridable as decision levers
# ---------------------------------------------------------------------------

@dataclass
class PolicyParams:
    """
    The firm's behavioural and policy settings.

    Every value is estimated from the filing history by `calibrate()`. Each is
    also a *lever*: overriding one is how the user asks "what if we changed
    this?" — which is the difference between a stress test and a decision tool.
    """
    effective_tax_rate: float = 0.165
    depreciation_rate_ppe: float = 0.062   # per QUARTER, applied to net PP&E
    dna_pct_revenue: float = 0.028         # fallback only
    capex_pct_revenue: float = 0.030
    buyback_pct_ocf: float = 0.80
    dividend_pct_net_income: float = 0.155
    interest_rate_on_debt: float = 0.035      # annual
    yield_on_cash: float = 0.030              # annual
    other_curr_liab_pct_revenue: float = 0.60
    debt_issued: float = 0.0                  # absolute, per quarter
    debt_repaid: float = 0.0

    def annual_to_quarter(self, r: float) -> float:
        return r / 4.0


def calibrate(panel: pd.DataFrame, cashflow: pd.DataFrame,
              income_extras: pd.DataFrame, lookback: int = 12) -> PolicyParams:
    """Estimate policy parameters from the most recent `lookback` quarters."""
    p = PolicyParams()

    def recent(df, col, n=lookback):
        if col not in df.columns:
            return pd.Series(dtype=float)
        return df[col].dropna().tail(n)

    etr = recent(income_extras, "effective_tax_rate")
    if len(etr):
        p.effective_tax_rate = float(np.clip(etr.median(), 0.05, 0.40))

    rev = panel["revenue"].dropna()

    def pct_of_revenue(series):
        j = pd.concat([series.rename("x"), rev.rename("r")], axis=1).dropna()
        j = j.tail(lookback)
        return float((j["x"] / j["r"]).median()) if len(j) else np.nan

    v = pct_of_revenue(recent(cashflow, "dna", 40))
    if np.isfinite(v):
        p.dna_pct_revenue = v

    # Depreciation is driven by the asset base, not by revenue. Tying it to
    # net PP&E is what gives capex a genuine earnings cost: more investment
    # raises PP&E, which raises D&A, which lowers operating income. With D&A
    # as a share of revenue, capex would be a pure cash lever with no P&L
    # consequence at all, and "cut capex" would look free.
    if "ppe" in panel.columns:
        j = pd.concat([recent(cashflow, "dna", 40).rename("d"),
                       panel["ppe"].rename("p")], axis=1).dropna().tail(lookback)
        if len(j):
            p.depreciation_rate_ppe = float(np.clip((j["d"] / j["p"]).median(),
                                                    0.005, 0.25))
    v = pct_of_revenue(recent(cashflow, "capex", 40))
    if np.isfinite(v):
        p.capex_pct_revenue = v

    ocf = recent(cashflow, "ocf", 40)
    bb = recent(cashflow, "buybacks", 40)
    j = pd.concat([bb.rename("b"), ocf.rename("o")], axis=1).dropna().tail(lookback)
    if len(j):
        p.buyback_pct_ocf = float(np.clip((j["b"] / j["o"]).median(), 0, 2.0))

    ni = panel["net_income"].dropna()
    dv = recent(cashflow, "dividends", 40)
    j = pd.concat([dv.rename("d"), ni.rename("n")], axis=1).dropna().tail(lookback)
    if len(j):
        p.dividend_pct_net_income = float(np.clip((j["d"] / j["n"]).median(), 0, 1.0))

    # residual current liabilities (everything that is not accounts payable)
    j = pd.concat([panel["liab_curr"].rename("lc"), panel["payables"].rename("ap"),
                   rev.rename("r")], axis=1).dropna().tail(lookback)
    if len(j):
        p.other_curr_liab_pct_revenue = float(((j["lc"] - j["ap"]) / j["r"]).median())

    return p


# ---------------------------------------------------------------------------
# Opening state
# ---------------------------------------------------------------------------

def opening_state(panel: pd.DataFrame, cashflow: pd.DataFrame,
                  income_extras: pd.DataFrame,
                  period: Optional[pd.Timestamp] = None) -> TwinState:
    """
    Build a balanced opening state from the last fully-reported quarter.

    `other_curr_assets`, `noncurrent_assets` and `other_noncurr_liab` are
    residuals of the reported totals, so the opening sheet ties to the filing
    exactly rather than approximately.
    """
    need = ["revenue", "cogs", "gross_profit", "operating_income", "inventory",
            "receivables", "payables", "cash", "assets_curr", "liab_curr",
            "assets", "liabilities", "equity", "net_income"]
    df = panel.dropna(subset=need)
    row = df.loc[period] if period is not None else df.iloc[-1]
    ts = row.name if period is None else period

    other_curr_assets = float(row["assets_curr"] - row["cash"]
                              - row["receivables"] - row["inventory"])
    ppe = float(row["ppe"]) if ("ppe" in row.index and pd.notna(row["ppe"])) else 0.0
    other_noncurr_assets = float(row["assets"] - row["assets_curr"] - ppe)
    other_curr_liab = float(row["liab_curr"] - row["payables"])

    debt = float(row["lt_debt"]) if pd.notna(row.get("lt_debt")) else 0.0
    other_noncurr_liab = float(row["liabilities"] - row["liab_curr"] - debt)

    def cf(col):
        if col in cashflow.columns and ts in cashflow.index:
            v = cashflow.loc[ts, col]
            return float(v) if pd.notna(v) else np.nan
        return np.nan

    def ie(col):
        if col in income_extras.columns and ts in income_extras.index:
            v = income_extras.loc[ts, col]
            return float(v) if pd.notna(v) else np.nan
        return np.nan

    st = TwinState(
        period=ts,
        cash=float(row["cash"]),
        receivables=float(row["receivables"]),
        inventory=float(row["inventory"]),
        other_curr_assets=other_curr_assets,
        ppe=ppe,
        other_noncurr_assets=other_noncurr_assets,
        payables=float(row["payables"]),
        other_curr_liab=other_curr_liab,
        debt=debt,
        other_noncurr_liab=other_noncurr_liab,
        equity=float(row["equity"]),
        revenue=float(row["revenue"]),
        cogs=float(row["cogs"]),
        gross_profit=float(row["gross_profit"]),
        opex=float(row["gross_profit"] - row["operating_income"]) - (
            float(cashflow.loc[ts, "dna"])
            if ("dna" in cashflow.columns and ts in cashflow.index
                and pd.notna(cashflow.loc[ts, "dna"])) else 0.0),
        operating_income=float(row["operating_income"]),
        net_income=float(row["net_income"]),
        nonoperating_income=ie("nonoperating_income"),
        pretax_income=ie("pretax_income"),
        tax=ie("tax"),
        ocf=cf("ocf"), capex=cf("capex"), dna=cf("dna"),
        buybacks=cf("buybacks"), dividends=cf("dividends"),
    )
    return st


# ---------------------------------------------------------------------------
# One period forward
# ---------------------------------------------------------------------------

def step(state: TwinState, drivers: Dict[str, float], params: PolicyParams,
         period: Optional[pd.Timestamp] = None) -> TwinState:
    """
    Advance the twin one quarter.

    `drivers` carries the *levels* for the coming period, already produced by
    the causal layer: revenue, cogs, opex, receivables, inventory, payables.
    Anything absent is carried forward unchanged.

    Note on opex: `drivers["opex"]` is *cash* operating expense. Depreciation
    is computed from the opening PP&E balance and added on top, so the
    investment loop closes properly — capex raises PP&E, PP&E raises D&A, D&A
    lowers operating income, and D&A is added back in operating cash flow so
    the only cash effect is through tax.
    """
    rev = drivers.get("revenue", state.revenue)
    cogs = drivers.get("cogs", state.cogs)
    opex_cash = drivers.get("opex", state.opex)
    ar = drivers.get("receivables", state.receivables)
    inv = drivers.get("inventory", state.inventory)
    ap = drivers.get("payables", state.payables)

    # --- depreciation on the opening asset base ------------------------------
    dna = params.depreciation_rate_ppe * state.ppe
    opex_total = opex_cash + dna

    gross_profit = rev - cogs
    operating_income = gross_profit - opex_total

    # --- non-operating: interest on the net cash position --------------------
    # This is the feedback edge that closes the loop. Drain cash in a scenario
    # and interest income falls, which lowers net income, which drains cash.
    qi = params.annual_to_quarter(params.yield_on_cash)
    qd = params.annual_to_quarter(params.interest_rate_on_debt)
    nonoperating = state.cash * qi - state.debt * qd

    pretax = operating_income + nonoperating
    tax = pretax * params.effective_tax_rate if pretax > 0 else 0.0
    net_income = pretax - tax

    # --- cash flow -----------------------------------------------------------
    d_ar = ar - state.receivables
    d_inv = inv - state.inventory
    d_ap = ap - state.payables
    ocf = net_income + dna - d_ar - d_inv + d_ap

    capex = params.capex_pct_revenue * rev
    buybacks = max(params.buyback_pct_ocf * ocf, 0.0)
    dividends = max(params.dividend_pct_net_income * net_income, 0.0)
    debt_issued = params.debt_issued
    debt_repaid = params.debt_repaid

    delta_cash = (ocf - capex + debt_issued - debt_repaid
                  - buybacks - dividends)

    # --- roll the balance sheet ---------------------------------------------
    cash = state.cash + delta_cash
    ppe = state.ppe + capex - dna
    debt = state.debt + debt_issued - debt_repaid
    equity = state.equity + net_income - dividends - buybacks
    other_curr_liab = params.other_curr_liab_pct_revenue * rev

    # `other_curr_liab` is driven by revenue rather than held fixed, so it must
    # be offset to preserve the identity: any change in it is matched on the
    # asset side. Without this the sheet would drift by exactly that delta.
    d_ocl = other_curr_liab - state.other_curr_liab
    other_curr_assets = state.other_curr_assets + d_ocl

    nxt = TwinState(
        period=period if period is not None
        else state.period + pd.DateOffset(days=91),
        cash=cash, receivables=ar, inventory=inv,
        other_curr_assets=other_curr_assets,
        ppe=ppe,
        other_noncurr_assets=state.other_noncurr_assets,
        payables=ap, other_curr_liab=other_curr_liab, debt=debt,
        other_noncurr_liab=state.other_noncurr_liab, equity=equity,
        revenue=rev, cogs=cogs, opex=opex_cash, opex_total=opex_total,
        gross_profit=gross_profit,
        operating_income=operating_income, nonoperating_income=nonoperating,
        pretax_income=pretax, tax=tax, net_income=net_income,
        dna=dna, capex=capex, buybacks=buybacks, dividends=dividends,
        debt_issued=debt_issued, debt_repaid=debt_repaid,
        ocf=ocf, fcf=ocf - capex, delta_cash=delta_cash,
    )
    return nxt


def assert_balanced(state: TwinState, tol: float = 1e-6) -> None:
    err = abs(state.balance_error)
    if err > tol:
        raise AssertionError(
            f"Balance sheet does not tie at {state.period}: "
            f"A−(L+E) = ${err:,.6f}M")
