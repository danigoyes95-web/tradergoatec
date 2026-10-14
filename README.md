# tradergoat

Toolkit propio para **investigar → validar → paper-trading** de estrategias sistemáticas, con control de riesgo estricto.
Núcleo pequeño (numpy + pandas), `ccxt` opcional para datos/exchanges.

```bash
pip install -e ".[dev]"
pytest                                             # 13 tests
tradergoat backtest    --synthetic --strategy donchian
tradergoat walkforward --symbol BTC/USDT --timeframe 1h --bars 8000 --strategy ema_trend
tradergoat goal        --symbol BTC/USDT --days 180 --start 100 --target 1000
tradergoat paper       --symbol BTC/USDT --strategy ema_trend --equity 100 --loop
```

## Qué hace cada pieza
| Módulo | Función | Inspirado en |
|---|---|---|
| `data.py` | OHLCV vía ccxt + caché CSV, validación, datos sintéticos | ccxt, freqtrade |
| `strategies.py` | Registro de estrategias (EMA trend, Donchian, RSI+filtro, breakout ATR) | freqtrade, backtrader |
| `backtest.py` | Backtest barra a barra, ejecución en t+1 (sin lookahead), comisiones y slippage | vectorbt, backtrader |
| `risk.py` | Vol-targeting, tope de apalancamiento, parada por drawdown, riesgo fijo por operación | hummingbot, lumibot |
| `walkforward.py` | Optimiza en train, puntúa SOLO en test (anti-overfitting) | freqtrade hyperopt |
| `montecarlo.py` | Probabilidad de llegar al objetivo vs. de arruinarse (bootstrap por bloques) | — |
| `paper.py` | Mismo código de señal/riesgo contra velas reales, estado atómico, sin órdenes reales | nautilus, lumibot |

Ver [`docs/ROADMAP.md`](docs/ROADMAP.md) para el plan 100$ → 1000$ y cómo integrar el resto de repos.

> Los números de `--synthetic` son solo para probar el código; no significan nada sobre rentabilidad.
