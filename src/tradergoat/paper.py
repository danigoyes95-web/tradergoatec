"""Paper trader: same signal/sizing/risk code as the backtest, executed against live candles with no real orders."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from . import strategies
from .risk import RiskConfig, position_size


@dataclass
class PaperState:
    equity: float = 100.0
    peak: float = 100.0
    position: float = 0.0        # leverage-signed exposure (notional / equity) held since last mark
    last_price: float = 0.0
    last_ts: str = ""
    cooldown: int = 0
    halted_count: int = 0

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(p)  # atomic: a crash never leaves a half-written state

    @classmethod
    def load(cls, path: str | Path, equity: float = 100.0) -> "PaperState":
        p = Path(path)
        if p.exists():
            return cls(**json.loads(p.read_text()))
        return cls(equity=equity, peak=equity)


def step(state: PaperState, df: pd.DataFrame, strategy: str, params: dict, cfg: RiskConfig) -> PaperState:
    """Process the newest *closed* candle in `df` (caller must drop the still-forming one). Idempotent per candle."""
    ts = str(df.index[-1])
    if ts == state.last_ts:
        return state
    price = float(df["close"].iloc[-1])
    if state.last_price > 0:  # mark the position held through this candle to market
        state.equity *= 1.0 + state.position * (price / state.last_price - 1.0)
    state.peak = max(state.peak, state.equity)
    if state.cooldown > 0:
        state.cooldown -= 1
    elif state.equity / state.peak - 1.0 < -cfg.max_drawdown:
        state.cooldown, state.peak = cfg.cooldown_bars, state.equity
        state.halted_count += 1
    sig = strategies.get(strategy)(df, **params).iloc[-1]
    target = 0.0 if state.cooldown > 0 else float(sig * position_size(df["close"], cfg).iloc[-1])
    state.equity *= 1.0 - cfg.cost * abs(target - state.position)
    state.position, state.last_price, state.last_ts = target, price, ts
    return state


def run_live(symbol: str, timeframe: str, strategy: str, params: dict, cfg: RiskConfig,
             state_path: str | Path, exchange: str = "binance", equity: float = 100.0,
             loop: bool = False, history: int = 600) -> PaperState:
    import ccxt

    from .data import validate

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    tf_s = ex.parse_timeframe(timeframe)
    state = PaperState.load(state_path, equity)
    while True:
        raw = ex.fetch_ohlcv(symbol, timeframe, limit=history)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
        df = validate(df).iloc[:-1]  # last candle is still forming
        state = step(state, df, strategy, params, cfg)
        state.save(state_path)
        print(f"{state.last_ts} eq={state.equity:.2f} pos={state.position:+.2f}x dd_halts={state.halted_count}")
        if not loop:
            return state
        time.sleep(tf_s - (time.time() % tf_s) + 5)  # wake just after the next candle closes
