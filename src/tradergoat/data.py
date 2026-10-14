"""OHLCV loading: ccxt (optional) with a CSV cache, plus a deterministic synthetic generator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def synthetic_ohlcv(n: int = 2000, seed: int = 0, freq: str = "1h",
                    drift: float = 0.00003, vol: float = 0.008, start: float = 100.0) -> pd.DataFrame:
    """Regime-switching random walk. Deterministic for a given seed; used by tests and demos."""
    rng = np.random.default_rng(seed)
    regime = np.cumsum(rng.random(n) < 0.004) % 2  # alternates trend / chop
    mu = np.where(regime == 0, drift * 6, -drift * 2)
    sig = np.where(regime == 0, vol, vol * 1.4)
    rets = mu + sig * rng.standard_normal(n)
    close = start * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start], close[:-1]])
    spread = np.abs(rng.standard_normal(n)) * sig * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    idx = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.random(n) * 1000 + 100}, index=idx)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Sort, drop duplicate timestamps and NaNs, and make sure columns are sane."""
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = df[COLUMNS].astype(float)
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna()
    if (out[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("non-positive prices in data")
    return out


def fetch_ccxt(symbol: str = "BTC/USDT", timeframe: str = "1h", limit: int = 5000,
               exchange: str = "binance", cache_dir: str | Path | None = "data") -> pd.DataFrame:
    """Download OHLCV through ccxt, paginating backwards-safe, and cache it as CSV."""
    import ccxt  # imported lazily so the core package works offline

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    since = ex.milliseconds() - limit * tf_ms
    rows: list[list[float]] = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= since:
            break
        since = nxt
    df = pd.DataFrame(rows, columns=["ts", *COLUMNS])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df = validate(df).tail(limit)
    if cache_dir:
        path = Path(cache_dir) / f"{exchange}_{symbol.replace('/', '-')}_{timeframe}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return validate(df)
