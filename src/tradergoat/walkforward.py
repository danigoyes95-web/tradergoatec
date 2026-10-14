"""Walk-forward optimisation: tune on a train window, score only on the unseen test window."""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from . import strategies
from .backtest import compute_stats, run_backtest
from .risk import RiskConfig


@dataclass
class WalkForwardResult:
    returns: pd.Series
    folds: list[dict]
    stats: dict


def grid(params: dict[str, list]) -> list[dict]:
    keys = list(params)
    return [dict(zip(keys, combo)) for combo in itertools.product(*params.values())]


def walk_forward(df: pd.DataFrame, strategy: str, params: dict[str, list], train: int, test: int,
                 cfg: RiskConfig | None = None, min_trades: int = 3) -> WalkForwardResult:
    cfg = cfg or RiskConfig()
    fn = strategies.get(strategy)
    combos = grid(params)
    if train <= 0 or test <= 0 or len(df) < train + test:
        raise ValueError("not enough data for one train+test fold")
    pieces, folds = [], []
    start = train
    while start + test <= len(df):
        tr = df.iloc[start - train:start]
        best, best_score = None, -1e18
        for p in combos:
            res = run_backtest(tr, fn(tr, **p), cfg)
            score = res.stats.get("sharpe", -1e18) if res.stats.get("trades", 0) >= min_trades else -1e18
            if score > best_score:
                best, best_score = p, score
        if best is None:  # nothing traded enough in train: stay flat for this fold
            best = combos[0]
            oos = pd.Series(0.0, index=df.index[start:start + test])
        else:
            seg = df.iloc[:start + test]  # causal indicators: history up to test end
            oos = run_backtest(seg, fn(seg, **best), cfg).returns.iloc[-test:]
        pieces.append(oos)
        folds.append({"start": df.index[start], "params": best, "train_sharpe": best_score,
                      "oos_return": float((1 + oos).prod() - 1)})
        start += test
    rets = pd.concat(pieces)
    pos = (rets != 0).astype(float)
    return WalkForwardResult(rets, folds, compute_stats(rets, pos, cfg.bars_per_year))
