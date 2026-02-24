# Algo Trading – Statistical Arbitrage (Pairs Trading)

## Overview

This project implements a **production-grade statistical arbitrage framework** based on
**monthly universe selection**, **pair eligibility screening**, and **walk-forward backtesting**.

The objective is to build a **robust, auditable, and scalable stat-arb engine** suitable for:
- research validation,
- portfolio construction,
- and eventual live deployment.

The framework has been stress-tested across multiple equity universes (France, Germany, Italy),
with consistent results after strict audit corrections.


---

## What Is Already Implemented ✅

### 1. Data Pipeline
- Daily OHLC data loading (CSV-based)
- Normalization using log-prices
- Caching mechanism for efficient batch processing
- Clean alignment of asset time series

---

### 2. Monthly Universe Construction
- Monthly universe snapshots (`trade_month`)
- Explicit `trade_start`, `trade_end`, and `scan_date`
- Fully parameterizable lookbacks
- Stored as parquet for reproducibility

---

### 3. Pair Scanner
- Exhaustive pair generation per universe
- Eligibility filters:
  - minimum data availability
  - stability checks
  - cointegration-style metrics
- Output:
  - eligibility flag
  - eligibility score
- Historical scanner fully implemented and audited

---

### 4. Monthly Batch Backtest (Core Engine)

Each month is treated as an **independent trading regime**:

- Monthly beta estimation (fixed for the whole month)
- Z-score based mean-reversion strategy
- Parameters:
  - `z_entry`, `z_exit`, `z_stop`
  - `z_window`
- Equal-weight portfolio construction
- Strict liquidation at month-end
- No carry-over by design (yet)

---

### 5. Trade Execution & Audit Trail (CRITICAL FIX)

The trade engine now correctly enforces:

- True trade entry date (`entry_datetime`)
- True exit date (`exit_datetime`)
- Trade must **enter within the trading month**
- Removal of warmup / lookahead artifacts
- Correct duration in days and bars

This fix reduced false trades dramatically (e.g. 348 → 38 trades),
confirming the engine’s correctness.

---

### 6. Global Walk-Forward Backtest

Implemented global aggregation across months:

- Equity chaining month by month
- Metrics computed on true global equity:
  - Final Equity
  - CAGR
  - Sharpe
  - Max Drawdown
  - Total number of trades
- Aggregated outputs:
  - global equity curve
  - monthly returns
  - full trade journal
  - pair-level metrics

---

### 7. Streamlit Interface

Interactive dashboard with tabs:

- Pair Monitor
- Scanner
- Monthly Universe
- Monthly Backtest
- Global Backtest (Walk-Forward)
- Pair Backtest
- Optimization (placeholder)
- Portfolio (placeholder)

All key backtests are executable end-to-end from the UI.

---

## Universes Tested So Far

| Universe | CAGR | Sharpe | Max DD | Trades |
|--------|------|--------|--------|--------|
| France | ~17% | ~3.5 | ~-2.6% | ~38 |
| Germany | ~20% | ~3.9 | ~-1.6% | ~43 |
| Italy | ~16% | ~2.3 | ~-4.7% | ~44 |

These results validate **cross-universe robustness**, while highlighting
market-quality differences (Italy acting as a natural stress test).

---

## Design Choices (Intentional)

- Monthly liquidation → avoids regime persistence bias
- Fixed beta per month → stable hedge ratio
- Equal-weight portfolio → transparency & robustness
- No optimization yet → avoids premature overfitting

---

## What Remains To Be Done 🚧

### 1. Position Capacity Constraint
- Select N eligible pairs per month (e.g. 20)
- Allow only K simultaneous positions (e.g. 10)
- Optional ranking of signals (instead of first-come)

Purpose:
- improve capital efficiency
- control risk concentration
- study return vs Sharpe trade-off

---

### 2. Multi-Universe Portfolio Aggregation
- Combine multiple universes (France, Germany, Italy, etc.)
- Allocation schemes:
  - equal-weight per universe
  - risk-weighted (volatility targeting)
- Produce pan-regional equity curve

---

### 3. Carry-Over Logic (Optional, Advanced)
- Conditional carry-over of open trades
- Risk-aware (only if spread converging)
- Explicit impact analysis on:
  - Sharpe
  - drawdowns
  - tail risk

---

### 4. Advanced Risk Controls
- Max exposure per pair
- Sector / country caps
- Volatility-adjusted sizing
- Gross / net exposure limits

---

### 5. Optimization Layer (Carefully)
- Parameter stability tests
- Cross-universe consistency checks
- Never optimize on a single universe

---

### 6. Production Readiness (Future)
- Live data feed integration
- Order simulation / slippage
- Event & earnings filters
- Execution constraints

---

## Key Takeaway

This project is **not a toy backtest**.

It is now:
- causally correct
- auditable
- robust across regimes
- extensible to large universes (1600+ assets)

Further improvements are about **portfolio construction and risk management**,
not fixing core logic.

---

## Author Notes

The framework is intentionally conservative.
Performance improvements should come from:
- better allocation,
- better signal selection,
- better diversification,
not from parameter overfitting.

---






## Utilisation des composants de robustesse (`utils/robustness.py`)

Tu peux appeler les composants directement dans ton notebook/script:

```python
from utils.robustness import (
    build_walk_forward_splits,
    pair_selection_stability,
    edge_decomposition,
)
```

### 1) Walk-forward splits

```python
splits = build_walk_forward_splits(
    start="2016-01-01",
    end="2025-12-31",
    train_months=24,
    test_months=6,
    step_months=3,
    anchored=False,   # True => anchored expanding
)
```

### 2) Stabilité des paires sélectionnées

```python
pair_stats, by_date = pair_selection_stability(scans_df, top_n=20)
```

- `pair_stats`: fréquence de sélection par paire
- `by_date`: turnover top-N et `topn_jaccard_vs_prev`

### 3) Edge decomposition

```python
edge = edge_decomposition(trades_df)
by_pair = edge["by_pair"]
by_side = edge["by_side"]
by_ym = edge["by_year_month"]
```

- utilise `trade_return_isolated` si disponible
- sinon fallback sur `trade_return`
