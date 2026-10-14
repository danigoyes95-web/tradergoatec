"""Bar-by-bar backtester: signals decided at close t are executed at t+1 (no lookahead)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk import RiskConfig, position_size


@dataclass
class Result:
    returns: pd.Series      # per-bar strategy returns, net of costs
    equity: pd.Series       # starts at 1.0
    position: pd.Series     # leverage-signed exposure actually held each bar
    stats: dict


def run_backtest(df: pd.DataFrame, signal: pd.Series, cfg: RiskConfig | None = None) -> Result:
    cfg = cfg or RiskConfig()
    sig = signal.reindex(df.index).fillna(0.0).clip(-1, 1)
    desired = (sig * position_size(df["close"], cfg)).to_numpy()
    ret = df["close"].pct_change().fillna(0.0).to_numpy()
    n = len(df)
    out = np.zeros(n)
    held = np.zeros(n)
    eq = peak = 1.0
    prev = 0.0
    cooldown = 0
    for t in range(n):
        target = desired[t - 1] if t > 0 else 0.0
        if cooldown > 0:
            target, cooldown = 0.0, cooldown - 1
        r = max(target * ret[t] - cfg.cost * abs(target - prev), -1.0)  # can't lose more than equity
        eq *= 1.0 + r
        peak = max(peak, eq)
        if eq <= 0:
            out[t:] = 0.0
            out[t] = r
            held[t] = target
            eq = 0.0
            break
        if eq / peak - 1.0 < -cfg.max_drawdown:
            cooldown, peak = cfg.cooldown_bars, eq
        out[t], held[t], prev = r, target, target
    returns = pd.Series(out, index=df.index)
    equity = (1.0 + returns).cumprod()
    position = pd.Series(held, index=df.index)
    return Result(returns, equity, position, compute_stats(returns, position, cfg.bars_per_year))


def compute_stats(returns: pd.Series, position: pd.Series, bars_per_year: float) -> dict:
    n = len(returns)
    if n < 2:
        return {}
    equity = (1.0 + returns).cumprod()
    years = n / bars_per_year
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    mu, sd = returns.mean(), returns.std()
    down = returns[returns < 0].std()
    mdd = float((equity / equity.cummax() - 1.0).min())
    entries = int(((position != 0) & (position.shift(1).fillna(0) == 0)).sum())
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": float(mu / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0,
        "sortino": float(mu / down * np.sqrt(bars_per_year)) if down and down > 0 else 0.0,
        "max_drawdown": mdd,
        "calmar": float(cagr / abs(mdd)) if mdd < 0 else 0.0,
        "trades": entries,
        "exposure": float((position != 0).mean()),
        "bars": n,
    }
