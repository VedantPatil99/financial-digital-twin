"""
Risk scoring for the Financial Digital Twin.

Scores are calibrated against the firm's *own* filing history rather than
textbook thresholds. A current ratio of 0.9 is alarming for a utility and
completely normal for Apple, which runs lean working capital by design. Fixed
industry cut-offs would flag Apple as permanently distressed and the dashboard
would be useless.

So each metric is scored by its percentile within that firm's history: 0 = the
healthiest this firm has ever been, 100 = the most stressed. That makes the
score answer the question a CFO actually asks -- "is this bad *for us*?"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# metric -> (direction, weight)
#   direction = -1 : higher is safer (liquidity, margins)
#   direction = +1 : higher is riskier (leverage)
RISK_METRICS = {
    "liquidity": {
        "current_ratio": (-1, 0.35),
        "quick_ratio": (-1, 0.35),
        "cash_ratio": (-1, 0.30),
    },
    "leverage": {
        "debt_to_equity": (+1, 0.40),
        "debt_to_assets": (+1, 0.40),
        "lt_debt_to_equity": (+1, 0.20),
    },
    "operational": {
        "operating_margin": (-1, 0.40),
        "gross_margin": (-1, 0.30),
        "asset_turnover": (-1, 0.30),
    },
}

CATEGORY_WEIGHTS = {"liquidity": 0.40, "leverage": 0.30, "operational": 0.30}

BANDS = [(0, 25, "Low"), (25, 50, "Moderate"), (50, 75, "Elevated"), (75, 101, "High")]


def band(score: float) -> str:
    if not np.isfinite(score):
        return "Unknown"
    for lo, hi, label in BANDS:
        if lo <= score < hi:
            return label
    return "High"


@dataclass
class RiskCalibration:
    """Empirical distribution of each metric over the firm's own history."""
    dists: Dict[str, np.ndarray]

    @classmethod
    def from_features(cls, feats: pd.DataFrame) -> "RiskCalibration":
        dists = {}
        for cat in RISK_METRICS.values():
            for metric in cat:
                if metric in feats.columns:
                    v = feats[metric].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(v) >= 8:
                        dists[metric] = v.to_numpy(float)
        return cls(dists)

    def percentile_score(self, metric: str, value: float, direction: int) -> float:
        """Map a value to 0-100 risk via its percentile in the firm's history."""
        d = self.dists.get(metric)
        if d is None or not np.isfinite(value):
            return np.nan
        pct = float((d < value).mean() * 100.0)
        # direction -1 (higher is safer) -> invert so high value = low risk
        return 100.0 - pct if direction < 0 else pct


def score_state(state: Dict[str, float], calib: RiskCalibration) -> Dict[str, object]:
    """
    Score a financial state (a dict of derived ratios) into risk categories.

    Returns per-category scores, the weighted composite, and per-metric detail.
    """
    detail, cat_scores = [], {}

    for cat, metrics in RISK_METRICS.items():
        num, den = 0.0, 0.0
        for metric, (direction, w) in metrics.items():
            val = state.get(metric, np.nan)
            s = calib.percentile_score(metric, val, direction)
            detail.append({"category": cat, "metric": metric, "value": val,
                           "risk_score": s, "weight": w})
            if np.isfinite(s):
                num += s * w
                den += w
        cat_scores[cat] = num / den if den > 0 else np.nan

    num = sum(cat_scores[c] * w for c, w in CATEGORY_WEIGHTS.items()
              if np.isfinite(cat_scores.get(c, np.nan)))
    den = sum(w for c, w in CATEGORY_WEIGHTS.items()
              if np.isfinite(cat_scores.get(c, np.nan)))
    composite = num / den if den > 0 else np.nan

    return {
        "categories": cat_scores,
        "composite": composite,
        "band": band(composite),
        "detail": pd.DataFrame(detail),
    }


def state_from_levels(levels: Dict[str, float]) -> Dict[str, float]:
    """Derive the ratio-space state the scorer expects from raw USD levels."""
    def d(a, b):
        a, b = levels.get(a, np.nan), levels.get(b, np.nan)
        return a / b if (np.isfinite(a) and np.isfinite(b) and b != 0) else np.nan

    out = {
        "current_ratio": d("assets_curr", "liab_curr"),
        "quick_ratio": ((levels.get("assets_curr", np.nan) - levels.get("inventory", 0.0))
                        / levels["liab_curr"]) if levels.get("liab_curr") else np.nan,
        "cash_ratio": d("cash", "liab_curr"),
        "debt_to_equity": d("liabilities", "equity"),
        "debt_to_assets": d("liabilities", "assets"),
        "lt_debt_to_equity": d("lt_debt", "equity"),
        "operating_margin": d("operating_income", "revenue"),
        "gross_margin": d("gross_profit", "revenue"),
        "asset_turnover": d("revenue", "assets"),
    }
    return out
