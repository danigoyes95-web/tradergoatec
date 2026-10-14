"""Answer 'how likely is 10x, and what does it cost in ruin risk?' from a strategy's own returns."""
from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np
import pandas as pd


def required_daily_growth(start: float, target: float, days: float) -> float:
    """Compound daily return needed to turn `start` into `target` in `days`."""
    return exp(log(target / start) / days) - 1.0


@dataclass
class GoalReport:
    p_target: float        # reached target before ruin
    p_ruin: float          # hit ruin level before target
    p_neither: float       # still in between at the horizon
    median_final: float
    p05_final: float
    p95_final: float


def goal_probability(returns: pd.Series | np.ndarray, start: float = 100.0, target: float = 1000.0,
                     ruin: float = 30.0, horizon_bars: int = 24 * 180, block: int = 24,
                     n_paths: int = 5000, scale: float = 1.0, seed: int = 0) -> GoalReport:
    """Block-bootstrap per-bar returns (keeps autocorrelation/vol clustering) into equity paths.

    `scale` multiplies returns to emulate more/less leverage (clipped so one bar can't lose >100%).
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < block * 2:
        raise ValueError("not enough return history to bootstrap")
    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon_bars // block)
    starts = rng.integers(0, len(r) - block, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon_bars]
    paths = np.cumprod(1.0 + np.clip(r[idx] * scale, -1.0, None), axis=1) * start
    hit_t = paths >= target
    hit_r = paths <= ruin
    first_t = np.where(hit_t.any(1), hit_t.argmax(1), horizon_bars + 1)
    first_r = np.where(hit_r.any(1), hit_r.argmax(1), horizon_bars + 1)
    win = first_t < first_r
    lose = first_r < first_t
    final = paths[:, -1]
    return GoalReport(float(win.mean()), float(lose.mean()), float((~win & ~lose).mean()),
                      float(np.median(final)), float(np.percentile(final, 5)), float(np.percentile(final, 95)))
