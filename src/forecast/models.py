"""
Forecasting engine for the Financial Digital Twin.

Predicts the twin's key flow variables (revenue, COGS, operating income) one or
more quarters ahead, with honest walk-forward validation.

Models
------
SeasonalNaive   y[t] = y[t-4].            The baseline that must be beaten.
DriftSeasonal   seasonal naive + fitted YoY drift.
RidgeLags       ridge regression on lagged log-levels + quarter dummies + macro.
GradientBoost   gradient-boosted trees on the same design matrix.
LSTMForecaster  small PyTorch LSTM over the log-level sequence.

A note on sample size: this panel has ~70 usable quarters. That is a lot of
*financial* history but very little by deep-learning standards, so the LSTM is
included and evaluated rather than assumed superior. The backtest decides.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEASON = 4  # quarters per fiscal year


# ---------------------------------------------------------------------------
# Design matrix
# ---------------------------------------------------------------------------

def make_supervised(y: pd.Series,
                    macro: Optional[pd.DataFrame] = None,
                    n_lags: int = 5,
                    fiscal_q: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build a supervised design matrix in log space.

    Working in logs makes the target multiplicative (financial series grow
    proportionally, not additively) and keeps errors comparable across the
    20x scale change from 2008 to 2026.
    """
    ly = np.log(y.where(y > 0)).dropna()
    X = pd.DataFrame(index=ly.index)

    for L in range(1, n_lags + 1):
        X[f"lag{L}"] = ly.shift(L)
    X["lag_season"] = ly.shift(SEASON)
    X["yoy_prev"] = ly.shift(SEASON) - ly.shift(SEASON * 2)
    X["mom_prev"] = ly.shift(1) - ly.shift(2)

    if fiscal_q is not None:
        fq = fiscal_q.reindex(ly.index)
        for q in (2, 3, 4):
            X[f"q{q}"] = (fq == q).astype(float)

    if macro is not None:
        for c in ["inflation_yoy", "gdp_yoy", "FEDFUNDS", "UNRATE"]:
            if c in macro.columns:
                X[c] = macro[c].reindex(ly.index).shift(1)  # lag: avoid lookahead

    df = pd.concat([X, ly.rename("__y")], axis=1).dropna()
    return df.drop(columns="__y"), df["__y"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BaseModel:
    name = "base"

    def fit(self, X, y, hist: pd.Series):
        return self

    def predict(self, X, hist: pd.Series) -> np.ndarray:
        raise NotImplementedError


class SeasonalNaive(BaseModel):
    name = "SeasonalNaive"

    def predict(self, X, hist):
        # log y[t-4] is already a column
        return X["lag_season"].to_numpy(float)


class DriftSeasonal(BaseModel):
    """Seasonal naive plus the average YoY log-drift seen in training."""
    name = "DriftSeasonal"

    def __init__(self):
        self.drift = 0.0

    def fit(self, X, y, hist):
        self.drift = float((y - X["lag_season"]).mean())
        return self

    def predict(self, X, hist):
        return X["lag_season"].to_numpy(float) + self.drift


class RidgeLags(BaseModel):
    name = "RidgeLags"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def fit(self, X, y, hist):
        Xs = self.scaler.fit_transform(X.to_numpy(float))
        self.model.fit(Xs, y.to_numpy(float))
        return self

    def predict(self, X, hist):
        return self.model.predict(self.scaler.transform(X.to_numpy(float)))


class GradientBoost(BaseModel):
    name = "GradientBoost"

    def __init__(self, **kw):
        self.model = GradientBoostingRegressor(
            n_estimators=kw.get("n_estimators", 300),
            max_depth=kw.get("max_depth", 2),
            learning_rate=kw.get("learning_rate", 0.05),
            subsample=0.9,
            random_state=0,
        )

    def fit(self, X, y, hist):
        self.model.fit(X.to_numpy(float), y.to_numpy(float))
        return self

    def predict(self, X, hist):
        return self.model.predict(X.to_numpy(float))


class LSTMForecaster(BaseModel):
    """
    Small LSTM over the log-level sequence.

    Deliberately tiny (hidden=16, 1 layer) with early stopping: with ~60
    training points anything larger memorises instead of generalising.
    Predicts the YoY log-change and adds it back to the seasonal lag, so the
    network learns the deviation from seasonality rather than the level.
    """
    name = "LSTM"

    def __init__(self, hidden: int = 16, epochs: int = 400, seq_len: int = 8,
                 lr: float = 0.01, seed: int = 0):
        self.hidden, self.epochs, self.seq_len, self.lr, self.seed = (
            hidden, epochs, seq_len, lr, seed)
        self.net = None
        self.mu = self.sd = None

    @staticmethod
    def _available() -> bool:
        # Broad except on purpose: a partially-installed torch raises OSError /
        # AttributeError rather than ImportError, and an unavailable optional
        # dependency must never take the whole backtest down.
        try:
            import torch  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def _seqs(self, feats: np.ndarray, target: np.ndarray):
        xs, ys = [], []
        for i in range(self.seq_len, len(feats)):
            xs.append(feats[i - self.seq_len:i])
            ys.append(target[i])
        return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)

    def fit(self, X, y, hist):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        resid = (y - X["lag_season"]).to_numpy(float)   # YoY log-change
        feats = X.to_numpy(float)
        self.mu, self.sd = feats.mean(0), feats.std(0) + 1e-8
        feats = (feats - self.mu) / self.sd

        xs, ys = self._seqs(feats, resid)
        if len(xs) < 12:
            self.net = None      # not enough sequence data; fall back
            return self

        n_val = max(3, int(len(xs) * 0.15))
        xtr, ytr = xs[:-n_val], ys[:-n_val]
        xva, yva = xs[-n_val:], ys[-n_val:]

        class Net(nn.Module):
            def __init__(self, n_in, hid):
                super().__init__()
                self.lstm = nn.LSTM(n_in, hid, batch_first=True)
                self.head = nn.Linear(hid, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        self.net = Net(xs.shape[2], self.hidden)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.MSELoss()

        Xtr = torch.tensor(xtr); Ytr = torch.tensor(ytr)
        Xva = torch.tensor(xva); Yva = torch.tensor(yva)

        best, best_state, patience, bad = np.inf, None, 60, 0
        for _ in range(self.epochs):
            self.net.train()
            opt.zero_grad()
            loss = lossf(self.net(Xtr), Ytr)
            loss.backward()
            opt.step()

            self.net.eval()
            with torch.no_grad():
                vl = float(lossf(self.net(Xva), Yva))
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state:
            self.net.load_state_dict(best_state)
        return self

    def predict(self, X, hist):
        seasonal = X["lag_season"].to_numpy(float)
        if self.net is None:
            return seasonal
        import torch
        feats = (X.to_numpy(float) - self.mu) / self.sd
        preds = np.zeros(len(feats))
        self.net.eval()
        for i in range(len(feats)):
            lo = max(0, i - self.seq_len + 1)
            window = feats[lo:i + 1]
            if len(window) < self.seq_len:  # left-pad early rows
                pad = np.repeat(window[:1], self.seq_len - len(window), axis=0)
                window = np.vstack([pad, window])
            with torch.no_grad():
                preds[i] = float(self.net(
                    torch.tensor(window[None, :, :], dtype=torch.float32)))
        return seasonal + preds


class SeasonalMomentum(BaseModel):
    """
    Seasonal naive scaled by *recent* YoY growth rather than full-sample drift.

    DriftSeasonal averages drift over all history, which drags an 8%-growth
    2025 toward the 40%-growth years of the early 2010s. Using only the last
    `window` observations tracks the current growth regime.
    """
    name = "SeasonalMomentum"

    def __init__(self, window: int = 4, damping: float = 0.7):
        self.window, self.damping, self.drift = window, damping, 0.0

    def fit(self, X, y, hist):
        recent = (y - X["lag_season"]).tail(self.window)
        self.drift = float(recent.mean()) * self.damping if len(recent) else 0.0
        return self

    def predict(self, X, hist):
        return X["lag_season"].to_numpy(float) + self.drift


class ResidualModel(BaseModel):
    """
    Wrap a scikit-learn regressor so it predicts the *deviation from
    seasonality* instead of the log level.

    This matters more than the choice of learner. The log level of Apple
    revenue rises ~20x across the sample, so a model targeting it spends its
    capacity tracking a trend that seasonal-naive already captures for free.
    The residual (y - y[t-4]) is roughly stationary, which is what these
    estimators are actually good at.
    """

    def __init__(self, estimator, name: str, scale: bool = True):
        self.estimator, self.name, self.scale = estimator, name, scale
        self.scaler = StandardScaler() if scale else None

    def fit(self, X, y, hist):
        resid = (y - X["lag_season"]).to_numpy(float)
        Xn = X.to_numpy(float)
        if self.scaler is not None:
            Xn = self.scaler.fit_transform(Xn)
        self.estimator.fit(Xn, resid)
        return self

    def predict(self, X, hist):
        Xn = X.to_numpy(float)
        if self.scaler is not None:
            Xn = self.scaler.transform(Xn)
        return X["lag_season"].to_numpy(float) + self.estimator.predict(Xn)


def all_models() -> List[BaseModel]:
    models = [
        SeasonalNaive(),
        DriftSeasonal(),
        SeasonalMomentum(window=4, damping=0.7),
        RidgeLags(alpha=1.0),
        GradientBoost(),
        ResidualModel(Ridge(alpha=3.0), "ResidualRidge"),
        ResidualModel(
            GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                      learning_rate=0.05, subsample=0.9,
                                      random_state=0),
            "ResidualGBM", scale=False),
    ]
    if LSTMForecaster._available():
        models.append(LSTMForecaster())
    return models
