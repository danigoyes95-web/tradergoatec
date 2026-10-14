"""Risk primitives shared by the backtester and the paper trader."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import realized_vol


@dataclass(frozen=True)
class RiskConfig:
    target_vol_annual: float = 0.60   # annualised volatility the book aims for
    max_leverage: float = 3.0         # hard cap on |position| / equity
    vol_window: int = 48              # bars used to estimate realised vol
    max_drawdown: float = 0.25        # halt when equity falls this far from its peak
    cooldown_bars: int = 72           # bars to stay flat after a halt
    fee: float = 0.0006               # per-side taker fee (fraction of notional)
    slippage: float = 0.0003          # per-side slippage (fraction of notional)
    bars_per_year: float = 24 * 365   # 1h crypto by default

    def __post_init__(self):
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown must be in (0, 1)")
        if self.max_leverage <= 0 or self.target_vol_annual <= 0:
            raise ValueError("max_leverage and target_vol_annual must be > 0")

    @property
    def cost(self) -> float:
        return self.fee + self.slippage


def position_size(close: pd.Series, cfg: RiskConfig) -> pd.Series:
    """Volatility-targeted leverage in [0, max_leverage] using only data up to each bar."""
    per_bar_target = cfg.target_vol_annual / np.sqrt(cfg.bars_per_year)
    vol = realized_vol(close, cfg.vol_window)
    size = (per_bar_target / vol.replace(0, np.nan)).clip(upper=cfg.max_leverage)
    return size.fillna(0.0)


def fixed_fractional_units(equity: float, risk_frac: float, entry: float, stop: float) -> float:
    """Units to buy so that hitting `stop` loses `risk_frac` of equity (classic 1% rule)."""
    dist = abs(entry - stop)
    if dist <= 0 or entry <= 0:
        return 0.0
    return equity * risk_frac / dist
