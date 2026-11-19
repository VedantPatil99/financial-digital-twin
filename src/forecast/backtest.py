"""
Walk-forward backtesting.

Why walk-forward and not a random train/test split: a random split lets the
model see 2025 while predicting 2019, which leaks the future and produces
accuracy numbers that cannot be reproduced in use. Here the model is refit at
every step on data available *strictly before* the predicted quarter -- the
same information a planner would actually have had.

Metrics are computed on the original USD scale (not logs), because a 2%
error on $400B revenue is what a decision-maker cares about.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))

from forecast.models import (all_models, make_supervised,  # noqa: E402
                             SeasonalNaive)


def _metrics(actual: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    ok = np.isfinite(a) & np.isfinite(p) & (a != 0)
    a, p = a[ok], p[ok]
    if len(a) == 0:
        return {"MAPE": np.nan, "RMSE": np.nan, "MAE": np.nan, "n": 0}
    err = p - a
    return {
        "MAPE": float(np.mean(np.abs(err / a)) * 100),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "n": int(len(a)),
    }


def walk_forward(y: pd.Series,
                 macro: pd.DataFrame,
                 fiscal_q: pd.Series,
                 min_train: int = 32,
                 models=None) -> pd.DataFrame:
    """
    Expanding-window backtest, one quarter ahead.

    Returns a tidy frame of (period, model, actual, predicted).
    """
    X_all, y_all = make_supervised(y, macro=macro, fiscal_q=fiscal_q)
    if len(X_all) <= min_train:
        raise ValueError(f"Not enough observations: {len(X_all)} <= {min_train}")

    rows = []
    for t in range(min_train, len(X_all)):
        Xtr, ytr = X_all.iloc[:t], y_all.iloc[:t]
        Xte, yte = X_all.iloc[t:t + 1], y_all.iloc[t:t + 1]
        period = X_all.index[t]

        # Fresh instances per fold so no state leaks across the expanding window.
        for model in (models if models is not None else all_models()):
            try:
                model.fit(Xtr, ytr, ytr)
                pred_log = float(np.asarray(model.predict(Xte, ytr))[0])
            except Exception as e:  # noqa: BLE001
                rows.append({"period": period, "model": model.name,
                             "actual": np.nan, "pred": np.nan, "error": str(e)})
                continue
            rows.append({
                "period": period,
                "model": model.name,
                "actual": float(np.exp(yte.iloc[0])),
                "pred": float(np.exp(pred_log)),
            })
    return pd.DataFrame(rows)


def summarise(bt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, grp in bt.groupby("model"):
        m = _metrics(grp["actual"].to_numpy(), grp["pred"].to_numpy())
        m["model"] = name
        out.append(m)
    df = pd.DataFrame(out).set_index("model").sort_values("MAPE")

    if "SeasonalNaive" in df.index:
        base = df.loc["SeasonalNaive", "MAPE"]
        df["skill_vs_naive_%"] = (base - df["MAPE"]) / base * 100
    return df


def main():
    from features import load_panel, build_features  # noqa: E402

    panel = load_panel()
    feats = build_features(panel)
    macro = pd.read_csv(os.path.join(ROOT, "data", "cache", "macro_quarterly.csv"),
                        index_col=0, parse_dates=True)

    results = {}
    for target in ["revenue", "cogs", "operating_income"]:
        y = panel[target].dropna()
        print(f"\n{'='*72}\nTarget: {target}   ({len(y)} quarters)\n{'='*72}")
        bt = walk_forward(y, macro, panel["fiscal_q"])
        tested = bt["period"].nunique()
        print(f"Walk-forward: {tested} out-of-sample quarters "
              f"({bt['period'].min().date()} -> {bt['period'].max().date()})\n")
        s = summarise(bt)
        print(s.round(3).to_string())
        results[target] = (bt, s)

        out = os.path.join(ROOT, "data", "cache", f"backtest_{target}.csv")
        bt.to_csv(out, index=False)

    print(f"\n{'='*72}\nInterpretation\n{'='*72}")
    rev_s = results["revenue"][1]
    best = rev_s.index[0]
    print(f"Best revenue model: {best} (MAPE {rev_s.iloc[0]['MAPE']:.2f}%)")
    if best == "SeasonalNaive":
        print("The seasonal baseline wins -- with ~70 quarters, the learned\n"
              "models do not have enough data to beat simple seasonality.")
    else:
        gain = rev_s.iloc[0].get("skill_vs_naive_%", np.nan)
        print(f"Beats the seasonal-naive baseline by {gain:.1f}% on MAPE.")
    return results


if __name__ == "__main__":
    main()
