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
      - PnL_total d'un trade = PnL_Y + PnL_X - fees
        avec :
          * Long spread  : +Y et -beta*X
          * Short spread : -Y et +beta*X
    """

    spread = spread.reset_index(drop=True)
    zscore = zscore.reset_index(drop=True)
    y = y.reset_index(drop=True)
    x = x.reset_index(drop=True)

    n = len(spread)
    equity = np.zeros(n)
    equity[0] = 1.0  # capital initial

    position = 0          # 0 = flat, +1 = long spread, -1 = short spread
    entry_i = None        # index d'entrée
    entry_equity = None   # niveau d'equity juste avant le trade

    trades: list[dict] = []
    pnl_list: list[float] = []

    for i in range(1, n):
        # Par défaut : equity = valeur de la veille
        equity[i] = equity[i - 1]

        # ============================================================
        #   Si FLAT -> chercher une entrée
        # ============================================================
        if position == 0:
            if zscore[i] > z_entry:
                # Short spread : -Y + beta*X
                position = -1
                entry_i = i
                entry_equity = equity[i]

            elif zscore[i] < -z_entry:
                # Long spread : +Y - beta*X
                position = 1
                entry_i = i
                entry_equity = equity[i]

        # ============================================================
        #   Si EN POSITION -> chercher sortie OU stop-loss
        # ============================================================
        else:
            if position == 1:  # LONG spread
                exit_signal = zscore[i] > -z_exit

            elif position == -1:  # SHORT spread
                exit_signal = zscore[i] < z_exit
            stop_signal = (
                z_stop is not None and
                abs(zscore[i]) >= z_stop
            )

            if exit_signal or stop_signal:
                # --- Calcul PnL du trade entier (de entry_i à i) ---
                dY = y[i] - y[entry_i]
                dX = x[i] - x[entry_i]

                # PnL par jambe
                pnl_y = position * dY
                pnl_x = -position * beta * dX
                pnl_gross = pnl_y + pnl_x

                trade_pnl = pnl_gross - fees

                # Mise à jour equity : on part de l'equity avant le trade
                equity[i] = entry_equity + trade_pnl

                trade = {
                    "Entry_index": int(entry_i),
                    "Exit_index": int(i),
                    "Side": "Long spread" if position == 1 else "Short spread",

                    # Spreads
                    "Entry_spread": float(spread[entry_i]),
                    "Exit_spread": float(spread[i]),

                    # Prix bruts
                    "Y_entry": float(y[entry_i]),
                    "Y_exit": float(y[i]),
                    "X_entry": float(x[entry_i]),
                    "X_exit": float(x[i]),

                    # Mouvements bruts
                    "dY": float(dY),
                    "dX": float(dX),

                    # Hedge ratio
                    "Beta": float(beta),

                    # PnL par jambe
                    "PnL_Y": float(pnl_y),
                    "PnL_X": float(pnl_x),

                    # PnL total
                    "PnL_total": float(trade_pnl),

                    # Durée en barres
                    "Duration": int(i - entry_i),

                    # Stop ?
                    "Stopped": bool(stop_signal),

                    # Winning
                    "Winning": bool(trade_pnl > 0),
                }

                trades.append(trade)
                pnl_list.append(trade_pnl)

                # Reset de la position
                position = 0
                entry_i = None
                entry_equity = None

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
    n = len(y)
    beta_wf = np.full(n, np.nan)
    spread_wf = np.full(n, np.nan)

    segments = walk_forward_segments(n, train, test)

    for (start_train, end_train, end_test) in segments:
        # 1) beta estimé sur TRAIN
        beta_seg = compute_hedge_ratio(y[start_train:end_train],
                                       x[start_train:end_train])

        # 2) spread pour le z-score : TRAIN + TEST
        spread_wf[start_train:end_test] = (
            y[start_train:end_test] - beta_seg * x[start_train:end_test]
        )

        # 3) beta pour la viz : TEST seulement
        beta_wf[end_train:end_test] = beta_seg

    # z-score sur le spread WF
    spread_series = pd.Series(spread_wf)
    zscore_wf = compute_zscore(spread_series, window=z_window)

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


