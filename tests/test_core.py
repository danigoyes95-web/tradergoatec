import numpy as np
import pandas as pd
import pytest

from tradergoat import strategies
from tradergoat.backtest import run_backtest
from tradergoat.data import synthetic_ohlcv, validate
from tradergoat.montecarlo import goal_probability, required_daily_growth
from tradergoat.paper import PaperState, step
from tradergoat.risk import RiskConfig, fixed_fractional_units
from tradergoat.walkforward import walk_forward


@pytest.fixture(scope="module")
def df():
    return synthetic_ohlcv(3000, seed=1)


def test_synthetic_is_deterministic_and_valid(df):
    assert df.equals(synthetic_ohlcv(3000, seed=1))
    validate(df)


@pytest.mark.parametrize("name", sorted(strategies.REGISTRY))
def test_strategies_bounded(df, name):
    s = strategies.get(name)(df)
    assert len(s) == len(df) and s.between(-1, 1).all() and not s.isna().any()


def test_no_lookahead(df):
    """Changing future prices must not change past strategy returns."""
    sig = strategies.get("ema_trend")(df)
    a = run_backtest(df, sig).returns
    d2 = df.copy()
    d2.iloc[-100:, :4] *= 1.5
    b = run_backtest(d2, strategies.get("ema_trend")(d2)).returns
    assert np.allclose(a.iloc[:-101], b.iloc[:-101])


def test_costs_reduce_returns(df):
    sig = strategies.get("ema_trend")(df)
    free = run_backtest(df, sig, RiskConfig(fee=0, slippage=0)).stats["total_return"]
    paid = run_backtest(df, sig, RiskConfig(fee=0.002, slippage=0.002)).stats["total_return"]
    assert paid < free


def test_leverage_cap_and_drawdown_halt():
    idx = pd.date_range("2024", periods=400, freq="1h", tz="UTC")
    close = pd.Series(100.0, index=idx)
    close.iloc[200:] = 100 * 0.97 ** np.arange(1, 201)  # relentless crash
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0})
    cfg = RiskConfig(max_leverage=2.0, max_drawdown=0.2, vol_window=10)
    res = run_backtest(df, pd.Series(1.0, index=idx), cfg)
    assert res.position.abs().max() <= 2.0 + 1e-9
    assert res.stats["max_drawdown"] > -0.6  # halt stops the bleeding vs. riding it to ~-98%
    assert res.equity.min() >= 0


def test_walk_forward_runs_oos_only(df):
    wf = walk_forward(df, "ema_trend", {"fast": [10, 20], "slow": [60, 120]}, train=1000, test=500)
    assert len(wf.folds) == 4 and len(wf.returns) == 2000


def test_goal_math():
    assert required_daily_growth(100, 1000, 180) == pytest.approx(0.01285, abs=1e-4)
    rng = np.random.default_rng(0)
    strong = pd.Series(rng.normal(0.002, 0.004, 5000))
    weak = pd.Series(rng.normal(-0.0002, 0.004, 5000))
    s = goal_probability(strong, horizon_bars=1500, n_paths=500)
    w = goal_probability(weak, horizon_bars=1500, n_paths=500)
    assert s.p_target > 0.9 and w.p_target < 0.05
    assert s.p_target + s.p_ruin + s.p_neither == pytest.approx(1.0)


def test_paper_step_idempotent_and_accounts_costs(df):
    cfg = RiskConfig()
    st = PaperState()
    st = step(st, df.iloc[:1500], "ema_trend", {}, cfg)
    snap = (st.equity, st.position)
    st = step(st, df.iloc[:1500], "ema_trend", {}, cfg)  # same candle again
    assert (st.equity, st.position) == snap
    st = step(st, df.iloc[:1501], "ema_trend", {}, cfg)
    assert st.equity > 0


def test_paper_state_roundtrip(tmp_path):
    p = tmp_path / "s.json"
    PaperState(equity=123.0, peak=130.0).save(p)
    assert PaperState.load(p).equity == 123.0


def test_fixed_fractional():
    assert fixed_fractional_units(100, 0.01, 50, 49) == pytest.approx(1.0)
    assert fixed_fractional_units(100, 0.01, 50, 50) == 0.0
