import numpy as np
import pandas as pd

from utils.metrics import compute_hedge_ratio, compute_zscore


# ============================================================
#   Simple & Correct Pairs Trading Backtest (beta-hedged)
#   avec Stop-Loss en Z-score
# ============================================================

def backtest_pair(
    spread: pd.Series,
    zscore: pd.Series,
    y: pd.Series,
    x: pd.Series,
    beta: float,
    beta_series: np.ndarray | pd.Series | None = None,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    z_stop: float | None = 4.0,
    fees: float = 0.0002,
):
    """
    Beta-hedged pairs trading backtest.

    Logique :
      - On ouvre un trade quand |zscore| dépasse z_entry.
      - On ferme quand |zscore| revient sous z_exit.
      - Stop-loss : si |zscore| >= z_stop (si z_stop n'est pas None).
      - Une seule position à la fois.

    Hedge:
      - Si beta_series est fourni (walk-forward), on "lock" le beta au moment de l'entrée
        et on le conserve jusqu'à la sortie.
      - Sinon, on utilise beta constant.

    IMPORTANT (Step 3):
      - On interprète le PnL de trade comme un RETURN (sur y/x normalisés log),
        et l'equity évolue multiplicativement:
            equity_exit = equity_entry * (1 + trade_return)

      - fees est un coût "round-trip" en return (ex: 0.0002 = 2 bps).
    """

    spread = spread.reset_index(drop=True)
    zscore = zscore.reset_index(drop=True)
    y = y.reset_index(drop=True)
    x = x.reset_index(drop=True)

    n = len(spread)
    if not (len(zscore) == len(y) == len(x) == n):
        raise ValueError("spread, zscore, y, x must have the same length")

    beta_arr = None
    if beta_series is not None:
        beta_arr = np.asarray(beta_series)
        if len(beta_arr) != n:
            raise ValueError("beta_series must have the same length as spread")

    equity = np.zeros(n, dtype=float)
    equity[0] = 1.0  # capital initial

    position = 0          # 0 = flat, +1 = long spread, -1 = short spread
    entry_i = None
    entry_equity = None
    beta_entry = None

    trades: list[dict] = []
    pnl_list: list[float] = []

    def _current_beta(i: int) -> float:
        if beta_arr is None:
            return float(beta)
        b = beta_arr[i]
        if np.isfinite(b):
            return float(b)
        return float(beta)

    for i in range(1, n):
        # Par défaut : equity = valeur de la veille
        equity[i] = equity[i - 1]

        # Si zscore non exploitable, on skip
        if not np.isfinite(zscore[i]):
            continue

        # ============================================================
        #   FLAT -> entrée
        # ============================================================
        if position == 0:
            b_now = _current_beta(i)
            if not np.isfinite(b_now):
                continue

            if zscore[i] > z_entry:
                position = -1  # short spread
                entry_i = i
                entry_equity = equity[i]
                beta_entry = b_now

            elif zscore[i] < -z_entry:
                position = 1   # long spread
                entry_i = i
                entry_equity = equity[i]
                beta_entry = b_now

        # ============================================================
        #   EN POSITION -> sortie / stop
        # ============================================================
        else:
            if position == 1:  # LONG spread
                exit_signal = zscore[i] > -z_exit
            else:              # SHORT spread
                exit_signal = zscore[i] < z_exit

            stop_signal = (z_stop is not None and abs(zscore[i]) >= z_stop)

            if exit_signal or stop_signal:
                dY = y[i] - y[entry_i]
                dX = x[i] - x[entry_i]

                b_used = float(beta_entry) if beta_entry is not None else float(beta)

                # Interprété comme return (sur séries log-normalisées)
                pnl_y = position * dY
                pnl_x = -position * b_used * dX
                pnl_gross = pnl_y + pnl_x

                trade_return = pnl_gross - fees  # return net du trade

                # Equity multiplicative
                new_equity = entry_equity * (1.0 + trade_return)

                # Sécurité : éviter equity <= 0 (sinon pct_change / métriques explosent)
                if not np.isfinite(new_equity) or new_equity <= 0:
                    new_equity = 1e-8

                equity[i] = new_equity

                trade = {
                    "Entry_index": int(entry_i),
                    "Exit_index": int(i),
                    "Side": "Long spread" if position == 1 else "Short spread",

                    "Entry_spread": float(spread[entry_i]) if np.isfinite(spread[entry_i]) else np.nan,
                    "Exit_spread": float(spread[i]) if np.isfinite(spread[i]) else np.nan,

                    "Y_entry": float(y[entry_i]),
                    "Y_exit": float(y[i]),
                    "X_entry": float(x[entry_i]),
                    "X_exit": float(x[i]),

                    "dY": float(dY),
                    "dX": float(dX),

                    "Beta_entry": float(b_used),

                    "PnL_Y": float(pnl_y),
                    "PnL_X": float(pnl_x),

                    # IMPORTANT: maintenant c'est un RETURN
                    "PnL_total": float(trade_return),

                    "Duration": int(i - entry_i),
                    "Stopped": bool(stop_signal),
                    "Winning": bool(trade_return > 0),
                }

                trades.append(trade)
                pnl_list.append(float(trade_return))

                # Reset
                position = 0
                entry_i = None
                entry_equity = None
                beta_entry = None

    equity = pd.Series(equity, name="equity")
    return equity, trades, pnl_list

# ============================================================
#   Performance Metrics
# ============================================================

def compute_metrics(equity: pd.Series, trades):
    """Compute Sharpe, Max Drawdown, CAGR, Total Return, Trade Count,
    Win rate, Avg Win/Loss, Max Cons Wins/Losses, Avg Trade Duration.
    """

    returns = equity.pct_change().fillna(0)

    # Sharpe
    sharpe = (
        np.sqrt(252) * returns.mean() / returns.std()
        if returns.std() > 0 else np.nan
    )

    # Max Drawdown
    max_dd = (equity / equity.cummax() - 1).min()

    # CAGR
    if len(equity) > 1:
        cagr = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
    else:
        cagr = np.nan

    # -------------------------------------------------
    # Trade-based metrics
    # -------------------------------------------------
    if trades and len(trades) > 0:

        nb_trades = len(trades)
        pnls = [t.get("PnL_total", np.nan) for t in trades]

        # Winrate
        winning_flags = [t.get("Winning", False) for t in trades]
        nb_wins = sum(winning_flags)
        winrate = nb_wins / nb_trades

        # Avg Win / Avg Loss
        winning_pnls = [p for p in pnls if p > 0]
        losing_pnls = [p for p in pnls if p < 0]

        avg_win = np.mean(winning_pnls) if winning_pnls else np.nan
        avg_loss = np.mean(losing_pnls) if losing_pnls else np.nan

        # Max Consecutive Wins / Losses
        max_consec_wins = 0
        max_consec_losses = 0

        cur_wins = 0
        cur_losses = 0

        for t in trades:
            if t.get("Winning", False):
                cur_wins += 1
                cur_losses = 0
            else:
                cur_losses += 1
                cur_wins = 0

            max_consec_wins = max(max_consec_wins, cur_wins)
            max_consec_losses = max(max_consec_losses, cur_losses)

        # Average Trade Duration
        durations = [t.get("Duration", np.nan) for t in trades]
        avg_duration = np.nanmean(durations)

    else:
        nb_trades = 0
        winrate = np.nan
        avg_win = np.nan
        avg_loss = np.nan
        max_consec_wins = 0
        max_consec_losses = 0
        avg_duration = np.nan

    return {
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "CAGR": cagr,
        "Final Equity": equity.iloc[-1],
        "Total Return": equity.iloc[-1] - 1,
        "Trades": nb_trades,
        "Winrate": winrate,
        "AvgWinningTrade": avg_win,
        "AvgLosingTrade": avg_loss,
        "MaxConsecWins": max_consec_wins,
        "MaxConsecLosses": max_consec_losses,
        "AvgTradeDuration": avg_duration,
    }




def walk_forward_segments(n, train, test):
    """
    Retourne une liste de (start_train, end_train, end_test)
    qui couvrent toute la série jusqu'à la fin.

    - train : longueur de la fenêtre d'estimation de beta
    - test  : longueur max de la fenêtre de trading (peut être plus courte sur le dernier bloc)
    """
    segments = []
    i = 0
    while i + train < n:
        start_train = i
        end_train = i + train
        end_test = min(end_train + test, n)   # dernier bloc éventuellement plus court
        segments.append((start_train, end_train, end_test))
        i += test
    return segments



def walk_forward_beta_spread_zscore(y, x, train, test, z_window):
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    n = len(y_arr)

    beta_wf = np.full(n, np.nan)
    spread_wf = np.full(n, np.nan)

    segments = walk_forward_segments(n, train, test)

    for (start_train, end_train, end_test) in segments:
        beta_seg = compute_hedge_ratio(
            y_arr[start_train:end_train],
            x_arr[start_train:end_train],
        )

        if not np.isfinite(beta_seg):
            prev = beta_wf[end_train - 1] if end_train - 1 >= 0 else np.nan
            beta_seg = float(prev) if np.isfinite(prev) else 1.0

        spread_wf[start_train:end_test] = (
            y_arr[start_train:end_test] - beta_seg * x_arr[start_train:end_test]
        )

        beta_wf[end_train:end_test] = beta_seg

    spread_series = pd.Series(spread_wf)
    zscore_wf = compute_zscore(spread_series, window=z_window).fillna(0.0)

    return spread_series, zscore_wf, beta_wf



def compute_segment_metrics(merged, trades, beta_wf, segments):
    report = []

    n = len(merged)

    for i, (train_start, train_end, test_end) in enumerate(segments):
        test_start = train_end

        idx_start = test_start
        idx_end = min(test_end, n)   # SECURE END INDEX

        segment_name = f"Seg {i+1}"

        # Dates (safe iloc)
        train_start_dt = merged["datetime"].iloc[train_start]
        train_end_dt   = merged["datetime"].iloc[min(train_end-1, n-1)]
        test_start_dt  = merged["datetime"].iloc[test_start]
        test_end_dt    = merged["datetime"].iloc[min(idx_end-1, n-1)]

        # Slice safe
        seg_pnl = merged["PnL_equity"].iloc[idx_start:idx_end]
        seg_beta = np.nanmean(beta_wf[idx_start:idx_end])

        # Trades in segment
        seg_trades = [t for t in trades if idx_start <= t["Entry_index"] < idx_end]

        if len(seg_trades) > 0:
            winrate = np.mean([t["Winning"] for t in seg_trades])
            avg_pnl = np.mean([t["PnL_total"] for t in seg_trades])
        else:
            winrate = np.nan
            avg_pnl = np.nan

        if len(seg_pnl) > 1:
            returns = seg_pnl.pct_change().dropna()
            sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else np.nan
        else:
            sharpe = np.nan

        max_dd = (seg_pnl / seg_pnl.cummax() - 1).min() if len(seg_pnl) else np.nan

        report.append({
            "Segment": segment_name,
            "Train Start": train_start_dt,
            "Train End": train_end_dt,
            "Test Start": test_start_dt,
            "Test End": test_end_dt,
            "Beta": seg_beta,
            "Trades": len(seg_trades),
            "Winrate": winrate,
            "AvgPnL": avg_pnl,
            "Sharpe": sharpe,
            "MaxDD": max_dd
        })

    return pd.DataFrame(report)


