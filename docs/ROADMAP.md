# Plan realista: 100 $ → 1000 $

## La verdad incómoda
- 10x en 6 meses exige **+1.29 %/día compuesto**; en 12 meses, +0.63 %/día. Fondos profesionales buscan ~30-60 %/año.
- Con 100 $ la única forma de acercarse es **apalancamiento alto = alta probabilidad de ruina**. No hay plazo garantizado.
- Las comisiones/spreads pesan más en cuentas pequeñas. Un backtest bonito casi nunca se reproduce en vivo.
- Nada de esto es asesoramiento financiero. Solo arriesga dinero que puedas perder entero.

## Camino recomendado (de menor a mayor riesgo)
1. **Datos reales + backtest** (`tradergoat fetch`, `backtest`) en BTC/ETH/SOL, varias estrategias.
2. **Walk-forward** (`walkforward`): descarta lo que solo funciona en-muestra.
3. **`goal`**: mira P(llegar a 1000) vs P(ruina) a distintos `--scale` (apalancamiento). Elige el punto donde la ruina sea tolerable, no el que maximiza el objetivo.
4. **Paper trading ≥ 4-8 semanas** (`paper --loop`). Compara con el backtest; si diverge, hay un bug o sobreajuste.
5. **Vivo con 100 $**, apalancamiento ≤ 2-3x, parada por drawdown activa. Reinvierte y **añade capital de ingresos propios**: es la palanca más fiable para llegar a 1000 $.

## Cómo encajan tus repos (integración futura, en orden de valor)
1. **ccxt** — ya integrado (datos y exchange).
2. **freqtrade** (GPL-3) — como referencia de hyperopt/ejecución; *no* se copia código por la licencia. Si se quiere ejecución en vivo robusta, usarlo como servicio aparte.
3. **vectorbt** — barridos masivos de parámetros (si hace falta más velocidad que el bucle actual).
4. **hummingbot** — market making / arbitraje: edge real en cuentas pequeñas es dudoso (fees, inventario).
5. **Polymarket CLI** — mercados de predicción: otra fuente de edge (probabilidades mal calibradas), módulo aparte.
6. **TradingAgents / FinRL** — señales LLM/RL: solo como *filtro* de las estrategias clásicas y siempre validado con walk-forward; son los más propensos a sobreajuste.
7. **backtrader / lumibot / nautilus** — redundantes con el núcleo actual; útiles solo para brokers de acciones (lumibot) o ejecución de baja latencia (nautilus).

## Reglas del repo
- Sin lookahead (test incluido), costes siempre activos, riesgo antes que retorno.
- Toda estrategia nueva: `@register`, test de acotación, y pasar walk-forward antes de paper.
- Claves API solo por variables de entorno (`.env` ignorado por git). Nunca con permiso de retiro.
