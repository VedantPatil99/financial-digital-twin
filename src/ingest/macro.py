"""
Macroeconomic drivers for the Financial Digital Twin (FRED, St. Louis Fed).

Series used
-----------
FEDFUNDS  Effective Federal Funds Rate      (monthly, %)   -> cost of debt
CPIAUCSL  CPI-U, All Items, SA              (monthly, idx) -> inflation
UNRATE    Civilian Unemployment Rate        (monthly, %)   -> demand proxy
DGS10     10-Year Treasury Constant Maturity(daily,   %)   -> long rates
GDPC1     Real GDP, chained 2017 dollars    (quarterly)    -> demand

Monthly/daily series are aggregated to Apple's *fiscal* quarters: for a period
ending on date D we average the observations falling in the ~3 months ending D.
Aligning macro to the firm's own fiscal calendar (not calendar quarters) is what
makes the regressions in the causal layer meaningful.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "data", "raw", "macro_raw.txt")
OUT = os.path.join(ROOT, "data", "cache", "macro_quarterly.csv")

FRED_SERIES = ["FEDFUNDS", "CPIAUCSL", "UNRATE", "DGS10", "GDPC1"]
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={id}&cosd=2006-01-01"


# --------------------------------------------------------------------------
# Live path (used when running on a machine with open internet)
# --------------------------------------------------------------------------
def fetch_live(series=FRED_SERIES) -> dict:
    """Download each FRED series directly. Requires outbound network access."""
    import requests
    out = {}
    for sid in series:
        r = requests.get(FRED_CSV.format(id=sid), timeout=60)
        r.raise_for_status()
        df = pd.read_csv(pd.io.common.StringIO(r.text))
        df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        out[sid] = df.dropna().set_index("date")["value"]
    return out


# --------------------------------------------------------------------------
# Cached path (snapshot captured from FRED, byte-exact)
# --------------------------------------------------------------------------
def parse_cached(path: str = RAW) -> dict:
    """Parse the captured FRED snapshot into {series_id: Series[monthly]}."""
    txt = open(path).read().strip()
    # The capture arrived JSON-escaped: newlines are literal backslash-n.
    txt = txt.replace("\\n", "\n").strip().strip('"')
    out = {}
    for block in txt.split("@@"):
        block = block.strip().strip('"').strip()
        if not block:
            continue
        lines = [l for l in block.split("\n") if l.strip()]
        header = lines[0]
        sid = header.split()[0].strip('"')
        if "ERR" in header or "EX" in header or len(lines) < 2:
            continue
        # Pick the payload by shape, not position: the capture can carry a
        # trailing tab-context footer after the data line.
        import re as _re
        cand = [l for l in lines[1:] if _re.match(r"^\s*\d{6}:", l)]
        if not cand:
            continue
        payload = max(cand, key=len).strip().strip('"')
        rec = {}
        for part in payload.split("|"):
            if ":" not in part:
                continue
            ym, v = part.split(":", 1)
            v = v.strip()
            if not v or v == "." or len(ym) != 6:
                continue
            try:
                rec[pd.Timestamp(f"{ym[:4]}-{ym[4:6]}-01")] = float(v)
            except ValueError:
                continue
        if rec:
            out[sid] = pd.Series(rec).sort_index()
    return out


def to_fiscal_quarters(monthly: dict, period_ends) -> pd.DataFrame:
    """
    Aggregate monthly macro series onto a firm's fiscal period-end dates.

    For period end D, average observations in (D - 3 months, D].
    """
    period_ends = pd.DatetimeIndex(period_ends).sort_values()
    frame = {}
    for sid, s in monthly.items():
        vals = []
        for d in period_ends:
            start = d - pd.DateOffset(months=3)
            window = s[(s.index > start) & (s.index <= d)]
            vals.append(window.mean() if len(window) else np.nan)
        frame[sid] = pd.Series(vals, index=period_ends)
    df = pd.DataFrame(frame)

    # Derived drivers the causal layer actually consumes.
    if "CPIAUCSL" in df:
        df["inflation_yoy"] = df["CPIAUCSL"].pct_change(4)
    if "GDPC1" in df:
        df["gdp_yoy"] = df["GDPC1"].pct_change(4)
    if "FEDFUNDS" in df:
        df["rate_change"] = df["FEDFUNDS"].diff()

    df.index.name = "period_end"
    return df


def build(period_ends, prefer_live: bool = False) -> pd.DataFrame:
    monthly = None
    if prefer_live:
        try:
            monthly = fetch_live()
        except Exception as e:  # noqa: BLE001
            print(f"  [macro] live fetch failed ({e}); using cached snapshot")
    if monthly is None:
        monthly = parse_cached()
    return to_fiscal_quarters(monthly, period_ends)


if __name__ == "__main__":
    panel = pd.read_csv(
        os.path.join(ROOT, "data", "cache", "apple_quarterly_panel.csv"),
        index_col=0, parse_dates=True)

    monthly = parse_cached()
    print("Parsed FRED series:")
    for sid, s in monthly.items():
        print(f"  {sid:<10} n={len(s):>5}  {s.index.min().date()} -> {s.index.max().date()}")

    macro = build(panel.index)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    macro.to_csv(OUT)
    print(f"\nAligned to {len(macro)} Apple fiscal quarters -> {OUT}")
    print("\nLatest 6 quarters:")
    print(macro.tail(6).round(3).to_string())

    print("\nSpot checks vs known macro history:")
    def at(datestr, col):
        idx = macro.index[macro.index.get_indexer([pd.Timestamp(datestr)], method="nearest")][0]
        return macro.loc[idx, col]
    checks = [
        ("ZIRP: fed funds < 0.5% in 2014", at("2014-06-28", "FEDFUNDS") < 0.5),
        ("2023 hiking cycle: fed funds > 4%", at("2023-07-01", "FEDFUNDS") > 4.0),
        ("2022 inflation spike > 6% YoY", at("2022-06-25", "inflation_yoy") > 0.06),
        ("COVID unemployment spike > 10%", at("2020-06-27", "UNRATE") > 10.0),
    ]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {name}")
