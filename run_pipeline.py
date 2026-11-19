"""
Rebuild the entire Financial Digital Twin from source data.

    python run_pipeline.py            # rebuild from the cached SEC snapshot
    python run_pipeline.py --live     # re-download from SEC EDGAR + FRED

--live requires outbound internet and hits primary sources directly; it is the
mode to use for the report, so every number traces to a filing rather than a
cache.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")

STEPS = [
    ("Build quarterly panel from SEC XBRL", [sys.executable, "src/build_panel.py"]),
    ("Recover quarterly cash flow from YTD", [sys.executable, "src/ingest/cashflow.py"]),
    ("Align FRED macro series",             [sys.executable, "src/ingest/macro.py"]),
    ("Derive twin state features",          [sys.executable, "src/features.py"]),
    ("Fit causal graph",                    [sys.executable, "src/graph/causal_graph.py"]),
    ("Validate shock propagation",          [sys.executable, "tests/test_propagation.py"]),
    ("Validate three-statement engine",     [sys.executable, "tests/test_statements.py"]),
    ("Walk-forward backtest",               [sys.executable, "src/forecast/backtest.py"]),
    ("Monte Carlo scenarios",               [sys.executable, "src/simulate/monte_carlo.py"]),
    ("Historical replay validation",        [sys.executable, "src/twin/replay.py"]),
    ("Twin drift / learning loop",          [sys.executable, "src/twin/learning.py"]),
]


def run(label, cmd, env):
    print(f"\n{'='*78}\n>> {label}\n{'='*78}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    dt = time.time() - t0
    status = "OK" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"-- {label}: {status} in {dt:.1f}s")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Re-download from SEC EDGAR before rebuilding")
    ap.add_argument("--ticker", default="AAPL")
    args = ap.parse_args()

    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")

    steps = list(STEPS)
    if args.live:
        steps.insert(0, (f"Download {args.ticker} live from SEC EDGAR",
                         [sys.executable, "src/ingest/sec_client.py",
                          "--ticker", args.ticker]))

    ok = True
    for label, cmd in steps:
        if not run(label, cmd, env):
            ok = False
            break

    print(f"\n{'='*78}")
    if ok:
        print("Pipeline complete.\n\nLaunch the dashboard with:\n"
              "    streamlit run dashboard/app.py")
    else:
        print("Pipeline stopped on failure — see the step output above.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
