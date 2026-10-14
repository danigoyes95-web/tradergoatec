from __future__ import annotations

import argparse
import json

from . import strategies
from .backtest import run_backtest
from .data import fetch_ccxt, load_csv, synthetic_ohlcv
from .montecarlo import goal_probability, required_daily_growth
from .paper import run_live
from .risk import RiskConfig
from .walkforward import walk_forward

DEFAULT_GRIDS = {
    "ema_trend": {"fast": [10, 20, 40], "slow": [60, 100, 200]},
    "donchian": {"lookback": [30, 55, 100], "exit_lookback": [10, 20]},
    "rsi_reversion": {"n": [7, 14], "low": [20, 30]},
    "vol_breakout": {"n": [12, 24, 48], "k": [1.0, 1.5, 2.0]},
}


def _data(a):
    if a.synthetic:
        return synthetic_ohlcv(n=a.bars, seed=a.seed)
    if a.csv:
        return load_csv(a.csv)
    return fetch_ccxt(a.symbol, a.timeframe, a.bars, a.exchange)


def _cfg(a) -> RiskConfig:
    return RiskConfig(target_vol_annual=a.vol, max_leverage=a.max_leverage, max_drawdown=a.max_dd)


def _fmt(stats: dict) -> str:
    pct = {"total_return", "cagr", "max_drawdown", "exposure"}
    return "\n".join(f"  {k:13s} {v:9.2%}" if k in pct else f"  {k:13s} {v:9.2f}" for k, v in stats.items())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tradergoat")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--strategy", default="ema_trend", choices=sorted(strategies.REGISTRY))
        sp.add_argument("--symbol", default="BTC/USDT")
        sp.add_argument("--timeframe", default="1h")
        sp.add_argument("--exchange", default="binance")
        sp.add_argument("--bars", type=int, default=5000)
        sp.add_argument("--csv")
        sp.add_argument("--synthetic", action="store_true", help="offline demo data")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--vol", type=float, default=0.60, help="annual target volatility")
        sp.add_argument("--max-leverage", type=float, default=3.0)
        sp.add_argument("--max-dd", type=float, default=0.25, help="halt trading at this drawdown")

    sp = sub.add_parser("backtest"); common(sp)
    sp.add_argument("--params", default="{}", help='JSON, e.g. \'{"fast":20,"slow":100}\'')
    sp = sub.add_parser("walkforward"); common(sp)
    sp.add_argument("--train", type=int, default=1500)
    sp.add_argument("--test", type=int, default=500)
    sp = sub.add_parser("goal"); common(sp)
    sp.add_argument("--start", type=float, default=100)
    sp.add_argument("--target", type=float, default=1000)
    sp.add_argument("--days", type=int, default=180)
    sp.add_argument("--scale", type=float, default=1.0, help="extra leverage multiplier on returns")
    sp = sub.add_parser("paper"); common(sp)
    sp.add_argument("--params", default="{}")
    sp.add_argument("--state", default="state/paper.json")
    sp.add_argument("--equity", type=float, default=100)
    sp.add_argument("--loop", action="store_true")
    sp = sub.add_parser("fetch"); common(sp)

    a = p.parse_args(argv)
    cfg = _cfg(a)
    if a.cmd == "fetch":
        df = fetch_ccxt(a.symbol, a.timeframe, a.bars, a.exchange)
        print(f"saved {len(df)} bars {df.index[0]} -> {df.index[-1]}")
    elif a.cmd == "backtest":
        df = _data(a)
        res = run_backtest(df, strategies.get(a.strategy)(df, **json.loads(a.params)), cfg)
        print(f"{a.strategy} on {len(df)} bars\n{_fmt(res.stats)}")
    elif a.cmd == "walkforward":
        df = _data(a)
        wf = walk_forward(df, a.strategy, DEFAULT_GRIDS[a.strategy], a.train, a.test, cfg)
        print(f"walk-forward {a.strategy}: {len(wf.folds)} folds (out-of-sample only)\n{_fmt(wf.stats)}")
    elif a.cmd == "goal":
        df = _data(a)
        # bootstrap out-of-sample returns only: in-sample curves overstate what a live account would see
        wf = walk_forward(df, a.strategy, DEFAULT_GRIDS[a.strategy], 1500, 500, cfg)
        returns = wf.returns
        bpd = 24 if a.timeframe == "1h" else 1
        rep = goal_probability(returns, a.start, a.target, ruin=a.start * 0.3,
                               horizon_bars=a.days * bpd, scale=a.scale)
        need = required_daily_growth(a.start, a.target, a.days)
        print(f"need {need:.2%}/day compounded for {a.days} days to turn {a.start:g} into {a.target:g}")
        print(f"P(reach target)={rep.p_target:.1%}  P(ruin<30%)={rep.p_ruin:.1%}  P(neither)={rep.p_neither:.1%}")
        print(f"final equity p05/median/p95 = {rep.p05_final:.0f} / {rep.median_final:.0f} / {rep.p95_final:.0f}")
    elif a.cmd == "paper":
        run_live(a.symbol, a.timeframe, a.strategy, json.loads(a.params), cfg, a.state,
                 a.exchange, a.equity, a.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
