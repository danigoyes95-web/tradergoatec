"""Strategies map OHLCV -> desired direction in [-1, 1]. Sizing and risk are applied elsewhere.

Add your own with @register("name"); params are plain kwargs so grid search can sweep them.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .indicators import atr, ema, rsi

REGISTRY: dict[str, Callable[..., pd.Series]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


@register("ema_trend")
def ema_trend(df: pd.DataFrame, fast: int = 20, slow: int = 100, long_only: bool = False) -> pd.Series:
    """Trend following: long when fast EMA > slow EMA."""
    if fast >= slow:
        return pd.Series(0.0, index=df.index)
    sig = np.sign(ema(df["close"], fast) - ema(df["close"], slow))
    return sig.clip(lower=0) if long_only else sig


@register("donchian")
def donchian(df: pd.DataFrame, lookback: int = 55, exit_lookback: int = 20, long_only: bool = False) -> pd.Series:
    """Turtle-style breakout with a shorter exit channel. State is carried forward."""
    hi = df["high"].rolling(lookback).max().shift(1)
    lo = df["low"].rolling(lookback).min().shift(1)
    xhi = df["high"].rolling(exit_lookback).max().shift(1)
    xlo = df["low"].rolling(exit_lookback).min().shift(1)
    c = df["close"]
    state = np.zeros(len(df))
    cur = 0.0
    for i, (price, h, l, eh, el) in enumerate(zip(c.values, hi.values, lo.values, xhi.values, xlo.values)):
        if np.isnan(h) or np.isnan(el):
            state[i] = 0.0
            continue
        if price > h:
            cur = 1.0
        elif price < l and not long_only:
            cur = -1.0
        elif (cur > 0 and price < el) or (cur < 0 and price > eh):
            cur = 0.0
        state[i] = cur
    return pd.Series(state, index=df.index)


@register("rsi_reversion")
def rsi_reversion(df: pd.DataFrame, n: int = 14, low: float = 30, high: float = 70,
                  trend_filter: int = 200, long_only: bool = True) -> pd.Series:
    """Buy oversold dips only while price is above a long EMA; exit when RSI mean-reverts to 50."""
    r = rsi(df["close"], n)
    up = df["close"] > ema(df["close"], trend_filter)
    state = np.zeros(len(df))
    cur = 0.0
    for i, (rv, u) in enumerate(zip(r.values, up.values)):
        if cur == 0.0:
            if rv < low and u:
                cur = 1.0
            elif rv > high and not u and not long_only:
                cur = -1.0
        elif (cur > 0 and rv >= 50) or (cur < 0 and rv <= 50):
            cur = 0.0
        state[i] = cur
    return pd.Series(state, index=df.index)


@register("vol_breakout")
def vol_breakout(df: pd.DataFrame, n: int = 24, k: float = 1.5, long_only: bool = False) -> pd.Series:
    """Go with the move when the close leaves an ATR band around its EMA."""
    mid = ema(df["close"], n)
    band = k * atr(df, n)
    sig = pd.Series(0.0, index=df.index)
    sig[df["close"] > mid + band] = 1.0
    if not long_only:
        sig[df["close"] < mid - band] = -1.0
    return sig.replace(0.0, np.nan).ffill().fillna(0.0)  # hold until opposite break


def get(name: str) -> Callable[..., pd.Series]:
    if name not in REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]
