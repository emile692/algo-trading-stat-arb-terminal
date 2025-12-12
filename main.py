import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import itertools
import plotly.express as px

from config.params import lookback_mapping
from utils.backtest import backtest_pair, compute_metrics, walk_forward_beta_spread_zscore, walk_forward_segments, \
    compute_segment_metrics
from utils.loader import load_price_csv
from utils.metrics import (
    compute_hedge_ratio,
    compute_spread,
    compute_adf,
    compute_corr,
    compute_coint,
    compute_zscore,
    compute_half_life,
)
from utils.optimization import run_full_optimization
from utils.synthetic import (
    generate_synthetic_paths,
    generate_ou_paths,
    calibrate_params_from_pair,
    simulate_cointegrated_assets,
)


PROJECT_PATH = Path(__file__).resolve().parents[0]
DATA_PATH = PROJECT_PATH / "data" / "raw"


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="StatArb Terminal", layout="wide")

# Tous les tickers disponibles
raw_data_list = sorted([file.stem for file in DATA_PATH.glob("*.csv")])


# ============================================================
# Définition des univers
# ============================================================
UNIVERSES = {
    "BIG TECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "ADBE", "INTC", "CSCO"],
    "FAANG": ["META", "AMZN", "AAPL", "NFLX", "GOOGL"],
    "MEGA CAP TECH": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    "ALL AVAILABLE": raw_data_list,
}


# ============================================================
# Cache CSV loading
# ============================================================
@st.cache_data
def cached_load_price(asset: str, data_path: Path) -> pd.DataFrame:
    return load_price_csv(asset, data_path)


# ============================================================
# Session state defaults (pour le bouton Load)
# ============================================================
if "asset1" not in st.session_state:
    st.session_state["asset1"] = raw_data_list[0]

if "asset2" not in st.session_state:
    st.session_state["asset2"] = raw_data_list[1]

if "go_to_monitor" not in st.session_state:
    st.session_state["go_to_monitor"] = False


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Settings")

    source = st.selectbox("Data Source", ["FMP", "Yahoo", "Binance"])

    # Valeurs par défaut basées sur session_state
    asset1 = st.selectbox("Asset 1", raw_data_list,
                          index=raw_data_list.index(st.session_state["asset1"]))
    asset2 = st.selectbox("Asset 2", raw_data_list,
                          index=raw_data_list.index(st.session_state["asset2"]))

    timeframe = st.selectbox("Timeframe", ["Hourly", "Daily"])

    lookback = st.selectbox("Lookback", list(lookback_mapping.keys()))
    lb = lookback_mapping[lookback]


# ============================================================
# TABS
# ============================================================
tab_monitor, tab_scanner, tab_backtest, tab_synth, tab_opt = st.tabs([
    "Pair Monitor", "Scanner", "Backtest Pair", "Synthetic Paths", "Optimization"
])



# Force auto-navigation vers Pair Monitor après un "Load"
if st.session_state["go_to_monitor"]:
    st.session_state["go_to_monitor"] = False
    tab_monitor.select()


# ============================================================
# TAB 1 : PAIR MONITOR
# ============================================================
with tab_monitor:
    st.subheader("Pair Monitor")

    with st.spinner("Loading data & computing metrics..."):

        # Load data
        df1 = cached_load_price(asset1, DATA_PATH)
        df2 = cached_load_price(asset2, DATA_PATH)

        df1 = df1.iloc[-lb:].copy()
        df2 = df2.iloc[-lb:].copy()

        # Log-prices
        df1["log"] = np.log(df1["close"])
        df2["log"] = np.log(df2["close"])

        df1["norm"] = df1["log"] - df1["log"].iloc[0]
        df2["norm"] = df2["log"] - df2["log"].iloc[0]

        df1["logret"] = df1["log"].diff()
        df2["logret"] = df2["log"].diff()

        merged = pd.merge(
            df1[["datetime", "norm", "logret"]],
            df2[["datetime", "norm", "logret"]],
            on="datetime",
            how="inner",
            suffixes=(f"_{asset1}", f"_{asset2}"),
        )

        series_y = merged[f"norm_{asset1}"]
        series_x = merged[f"norm_{asset2}"]

        beta = compute_hedge_ratio(series_y, series_x)
        spread = compute_spread(series_y, series_x, beta)
        adf_t, adf_p, _ = compute_adf(spread.dropna())
        corr = compute_corr(series_y, series_x)
        eg_t, eg_p, _ = compute_coint(series_y, series_x)
        half_life = compute_half_life(spread)

        # Paramètres OU
        if half_life and half_life > 0:
            theta_ou = np.log(2) / half_life
        else:
            theta_ou = 0

        mu_ou = float(spread.mean())
        sigma_ou = float(np.std(spread.diff().dropna()))
        s0_ou = float(spread.iloc[-1])

        # Stockage pour l'onglet Synthetic Paths
        st.session_state["ou_mu"] = mu_ou
        st.session_state["ou_theta"] = theta_ou
        st.session_state["ou_sigma"] = sigma_ou
        st.session_state["ou_s0"] = s0_ou

        merged["spread"] = spread
        merged["zscore"] = compute_zscore(spread, window=30)
        merged["rolling_corr"] = series_y.rolling(200).corr(series_x)

        merged["cumret_y"] = np.exp(merged[f"logret_{asset1}"].fillna(0).cumsum()) - 1
        merged["cumret_x"] = np.exp(merged[f"logret_{asset2}"].fillna(0).cumsum()) - 1


    # METRICS
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Hedge Ratio", f"{beta:.3f}")
    col2.metric("ADF t-stat", f"{adf_t:.3f}")
    col3.metric("ADF p-value", f"{adf_p:.4f}")
    col4.metric("Correlation", f"{corr:.3f}")
    col5.metric("Stationarity?", "Yes" if adf_p < 0.05 else "No")
    col6.metric("Cointegrated?", "Yes" if eg_p < 0.05 else "No")
    col7.metric("Half-life", "N/A" if half_life is None else f"{half_life:.1f} bars")

    # =========================
    #  FIGURES (VERSION PRO)
    # =========================

    # --- Styling global ---
    default_layout = dict(
        height=250,
        template="plotly_dark",
        margin=dict(l=5, r=5, t=5, b=5),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )

    # ========== RETURNS ==========
    fig_returns = go.Figure()
    fig_returns.add_scatter(
        x=merged["datetime"], y=merged["cumret_y"],
        name=f"{asset1} cumret", line=dict(width=2)
    )
    fig_returns.add_scatter(
        x=merged["datetime"], y=merged["cumret_x"],
        name=f"{asset2} cumret", line=dict(width=2)
    )
    fig_returns.update_layout(**default_layout)

    # ========== SPREAD ==========
    # Couleurs dynamiques
    spread_pos = merged["spread"].where(merged["spread"] >= 0)
    spread_neg = merged["spread"].where(merged["spread"] < 0)

    fig_spread = go.Figure()

    # Spread positif
    fig_spread.add_scatter(
        x=merged["datetime"], y=spread_pos,
        mode="lines",
        name="Spread +",
        line=dict(color="#00cc96", width=2),
    )
    fig_spread.add_scatter(
        x=merged["datetime"], y=spread_pos,
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(0,204,150,0.15)",
        showlegend=False,
    )

    # Spread négatif
    fig_spread.add_scatter(
        x=merged["datetime"], y=spread_neg,
        mode="lines",
        name="Spread -",
        line=dict(color="#ff4d4d", width=2),
    )
    fig_spread.add_scatter(
        x=merged["datetime"], y=spread_neg,
        fill="tozeroy",
        mode="none",
        fillcolor="rgba(255,77,77,0.15)",
        showlegend=False,
    )

    # Ligne 0
    fig_spread.add_hline(y=0, line_color="white", opacity=0.3)

    fig_spread.update_layout(**default_layout)

    # ========== Z-SCORE ==========
    fig_z = go.Figure()
    fig_z.add_scatter(
        x=merged["datetime"], y=merged["zscore"],
        mode="lines", name="Z-score",
        line=dict(width=2, color="#1f77b4")
    )
    fig_z.add_hline(y=2, line_dash="dot", line_color="red")
    fig_z.add_hline(y=-2, line_dash="dot", line_color="green")
    fig_z.update_layout(**default_layout)

    # ========== ROLLING CORR ==========
    fig_corr = go.Figure()
    fig_corr.add_scatter(
        x=merged["datetime"], y=merged["rolling_corr"],
        mode="lines", name="Rolling Corr",
        line=dict(width=2, color="#ab63fa")
    )
    fig_corr.update_layout(**default_layout)

    # ========== SCATTER + OLS ==========
    fig_scatter = go.Figure()
    fig_scatter.add_scatter(
        x=series_x, y=series_y,
        mode="markers",
        name="points",
        marker=dict(size=5)
    )

    x_line = np.linspace(series_x.min(), series_x.max(), 100)
    y_line = beta * x_line + (series_y.mean() - beta * series_x.mean())

    fig_scatter.add_scatter(
        x=x_line, y=y_line,
        mode="lines",
        name=f"OLS β={beta:.2f}",
        line=dict(width=2, color="orange")
    )

    fig_scatter.update_layout(**default_layout)

    # =========================
    #  DISPLAY LAYOUT
    # =========================
    colA1, colA2, colA3 = st.columns(3)
    colB1, colB2, colB3 = st.columns(3)

    colA1.plotly_chart(fig_returns, width='stretch')
    colA2.plotly_chart(fig_spread, width='stretch')
    colA3.plotly_chart(fig_scatter, width='stretch')

    colB1.plotly_chart(fig_z, width='stretch')
    colB2.plotly_chart(fig_corr, width='stretch')

# ============================================================
# TAB 2 : PAIR SCREENER
# ============================================================
with tab_scanner:

    st.subheader("Pair Screener")

    universe_name = st.selectbox("Universe", list(UNIVERSES.keys()))
    scan_lb = lookback_mapping[st.selectbox("Lookback (screener)", list(lookback_mapping.keys()), index=3)]

    min_corr = st.slider("Min absolute correlation", 0.0, 1.0, 0.5, 0.05)
    max_half_life = st.slider("Max half-life (bars)", 5, 500, 150, 5)

    run = st.button("Run scan")

    if run:
        with st.spinner("Scanning..."):

            tickers = [t for t in UNIVERSES[universe_name] if t in raw_data_list]

            prepared = {}
            for t in tickers:
                df = cached_load_price(t, DATA_PATH).iloc[-scan_lb:].copy()
                df["log"] = np.log(df["close"])
                df["norm"] = df["log"] - df["log"].iloc[0]
                prepared[t] = df[["datetime", "norm"]]

            results = []

            for a1, a2 in itertools.combinations(prepared.keys(), 2):
                df1 = prepared[a1]
                df2 = prepared[a2]

                merged_s = pd.merge(
                    df1, df2, on="datetime", how="inner",
                    suffixes=(f"_{a1}", f"_{a2}")
                )

                if len(merged_s) < 50:
                    continue

                y = merged_s[f"norm_{a1}"]
                x = merged_s[f"norm_{a2}"]

                try:
                    beta_s = compute_hedge_ratio(y, x)
                    spread_s = compute_spread(y, x, beta_s)
                    adf_t_s, adf_p_s, _ = compute_adf(spread_s.dropna())
                    corr_s = compute_corr(y, x)
                    eg_t_s, eg_p_s, _ = compute_coint(y, x)
                    hl_s = compute_half_life(spread_s)
                except Exception:
                    continue

                if hl_s is None or hl_s <= 0:
                    continue

                if abs(corr_s) < min_corr or hl_s > max_half_life:
                    continue

                z_s = compute_zscore(spread_s)
                z_std = float(np.nanstd(z_s))

                ou_score = (1 - min(adf_p_s, 1.0)) / np.log1p(hl_s)
                score = corr_s * ou_score

                results.append({
                    "Asset1": a1,
                    "Asset2": a2,
                    "Corr": corr_s,
                    "ADF_p": adf_p_s,
                    "EG_p": eg_p_s,
                    "Half life": hl_s,
                    "Zscore std": z_std,
                    "OU Score": ou_score,
                    "Score": score
                })

            if not results:
                st.info("No pairs matched your filters.")
            else:
                df_res = pd.DataFrame(results).sort_values("Score", ascending=False)

                st.markdown("### Ranked pairs")

                # ====== En-têtes de tableau ======
                h1, h2, h3, h4, h5, h6, h7, h8, h9, h10 = st.columns(
                    [2, 2, 1, 1, 1, 1, 1, 1, 1.2, 1]
                )

                h1.write("Asset 1")
                h2.write("Asset 2")
                h3.write("Corr")
                h4.write("ADF p")
                h5.write("EG p")
                h6.write("HL")
                h7.write("Zscore std")
                h8.write("OU Score")
                h9.write("Score")
                h10.write("Load")

                # ====== Lignes du tableau ======
                for idx, row in df_res.iterrows():

                    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = st.columns(
                        [2, 2, 1, 1, 1, 1, 1, 1, 1.2, 1]
                    )

                    c1.write(row["Asset1"])
                    c2.write(row["Asset2"])
                    c3.write(f"{row['Corr']:.3f}")
                    c4.write(f"{row['ADF_p']:.4f}")
                    c5.write(f"{row['EG_p']:.4f}")
                    c6.write(f"{row['Half life']:.1f}")
                    c7.write(f"{row['Zscore std']:.3f}")
                    c8.write(f"{row['OU Score']:.3f}")
                    c9.write(f"{row['Score']:.3f}")

                    # ---- Bouton Load ----
                    if c10.button("Load", key=f"load_{idx}"):
                        st.session_state["asset1"] = row["Asset1"]
                        st.session_state["asset2"] = row["Asset2"]
                        st.session_state["go_to_monitor"] = True
                        st.experimental_rerun()

# ============================================================
# TAB 3 : BACKTEST PAIR
# ============================================================

with tab_backtest:

    st.subheader("Backtest Pair Trading")

    st.markdown("Réglages des paramètres de statégie")
    bc1, bc2, bc3, bc4, bc5 = st.columns(5)

    z_entry = bc1.slider("Z-entry", 1.0, 4.0, 2.0, 0.1)
    z_exit = bc2.slider("Z-exit", 0.0, 2.0, 0.4, 0.05)
    z_stop = bc3.slider("Z-stop (|z|)", 2.0, 8.0, 4.0, 0.1)
    z_window = bc4.slider("Z-window", 20, 200, 60, 10)
    fees = bc5.number_input("Fees (round trip)", value=0.0002, format="%.6f")

    st.markdown("Réglages du walk-forward")
    wf1, wf2 = st.columns(2)
    wf_train = wf1.slider("WF Train Window", 50, 500, 120, 10)
    wf_test = wf2.slider("WF Test Window", 10, 200, 30, 10)

    # ------------------------------------------------------------
    #  AUTO-RUN DU BACKTEST (pas de bouton)
    # ------------------------------------------------------------

    st.subheader(f"Backtest results for {asset1} / {asset2}")

    with st.spinner("Running backtest..."):

        # --- WALK-FORWARD SPREAD & ZSCORE ---
        spread_bt, zscore_bt, beta_wf = walk_forward_beta_spread_zscore(
            merged[f"norm_{asset1}"],
            merged[f"norm_{asset2}"],
            train=wf_train,
            test=wf_test,
            z_window=z_window,
        )

        equity, trades, pnl_list = backtest_pair(
            spread_bt,
            zscore_bt,
            merged[f"norm_{asset1}"],
            merged[f"norm_{asset2}"],
            beta,
            z_entry=z_entry,
            z_exit=z_exit,
            z_stop=z_stop,
            fees=fees,
        )

        # Inject equity into merged for WF segment analytics
        merged["PnL_equity"] = equity.values

        segments = walk_forward_segments(len(merged), wf_train, wf_test)

        metrics = compute_metrics(equity, trades)

    # ============================================================
    # Equity Curve
    # ============================================================
    fig_eq = go.Figure()
    fig_eq.add_scatter(
        x=merged["datetime"],
        y=equity,
        mode="lines",
        line=dict(width=2, color="#00c3ff"),
        name="Equity"
    )
    fig_eq.update_layout(
        height=350,
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig_eq, use_container_width=True)

    # ============================================================
    # METRICS
    # ============================================================

    m1, m2, m3, m4, m5 = st.columns(5)
    m6, m7, m8, m9, m10, m11 = st.columns(6)

    m1.metric("Final Equity", f"{metrics['Final Equity']:.3f}")
    m2.metric("Total Return", f"{metrics['Total Return'] * 100:.1f}%")
    m3.metric("Sharpe", f"{metrics['Sharpe']:.2f}")
    m4.metric("Max DD", f"{metrics['Max Drawdown'] * 100:.1f}%")

    m5.metric("Nb Trades", f"{metrics['Trades']}")
    winrate_display = f"{metrics['Winrate'] * 100:.1f}%" if metrics['Trades'] > 0 else "N/A"
    m6.metric("Winrate", winrate_display)

    avg_win_disp = f"{metrics['AvgWinningTrade']* 100:.2f}%" if not np.isnan(metrics['AvgWinningTrade']) else "N/A"
    avg_loss_disp = f"{metrics['AvgLosingTrade']* 100:.2f}%" if not np.isnan(metrics['AvgLosingTrade']) else "N/A"

    m7.metric("Avg Winning Trade", avg_win_disp)
    m8.metric("Avg Losing Trade", avg_loss_disp)

    m9.metric("Max Consecutive Wins", f"{metrics['MaxConsecWins']}")
    m10.metric("Max Consecutive Losses", f"{metrics['MaxConsecLosses']}")

    m11.metric("Avg Trade Duration", f"{metrics['AvgTradeDuration']:.1f} bars")

    # ============================================================
    # Trade List
    # ============================================================
    st.markdown("### Trade List")

    if len(trades) == 0:
        st.info("No trades taken with these parameters.")
    else:
        df_trades = pd.DataFrame(trades)
        df_trades.index.name = "Trade #"

        df_trades["PnL_total"] = df_trades["PnL_total"].round(5)
        df_trades["PnL_Y"] = df_trades["PnL_Y"].round(5)
        df_trades["PnL_X"] = df_trades["PnL_X"].round(5)
        df_trades["Duration"] = df_trades["Duration"].astype(int)

        st.dataframe(df_trades, use_container_width=True)

        # ============================================================
        # TRADE VISUALIZATION (Spread WF + classic spread background)
        # ============================================================
        st.markdown("### Trade Visualization")

        fig_trades = go.Figure()

        # ---- Ajout du spread classique (gris, discret) ----
        fig_trades.add_scatter(
            x=merged["datetime"],
            y=merged["spread"],
            mode="lines",
            name="Spread (classic)",
            line=dict(color="rgba(150,150,150,0.25)", width=1),
            showlegend=True
        )

        # ---- Spread WF (principal) ----
        fig_trades.add_scatter(
            x=merged["datetime"],
            y=spread_bt,
            mode="lines",
            name="Spread (WF)",
            line=dict(color="#00c3ff", width=2),
            showlegend=True
        )

        # ---- Markers entrées/sorties ----
        for t in trades:
            entry_i = t["Entry_index"]
            exit_i = t["Exit_index"]
            side = t["Side"]

            entry_color = "#00ff00" if "Long" in side else "#ff3300"
            exit_color = "white" if not t["Stopped"] else "#ff6600"

            fig_trades.add_scatter(
                x=[merged["datetime"].iloc[entry_i]],
                y=[spread_bt.iloc[entry_i]],
                mode="markers",
                marker=dict(size=10, color=entry_color),
                showlegend=False
            )

            fig_trades.add_scatter(
                x=[merged["datetime"].iloc[exit_i]],
                y=[spread_bt.iloc[exit_i]],
                mode="markers",
                marker=dict(size=10, color=exit_color, symbol="circle-open"),
                showlegend=False
            )

        fig_trades.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Spread"
        )

        st.plotly_chart(fig_trades, use_container_width=True)

        # ============================================================
        # Z-SCORE VISUALIZATION
        # ============================================================
        st.markdown("### Z-Score Confirmation")

        fig_ztrades = go.Figure()

        fig_ztrades.add_scatter(
            x=merged["datetime"],
            y=zscore_bt,
            mode="lines",
            name="Z-score",
            line=dict(color="#ffaa00", width=2),
        )

        fig_ztrades.add_hline(y=z_entry, line_dash="dot", line_color="#ff6600")
        fig_ztrades.add_hline(y=-z_entry, line_dash="dot", line_color="#ff6600")
        fig_ztrades.add_hline(y=z_exit, line_dash="dot", line_color="white", opacity=0.3)
        fig_ztrades.add_hline(y=-z_exit, line_dash="dot", line_color="white", opacity=0.3)
        fig_ztrades.add_hline(y=z_stop, line_dash="dot", line_color="red")
        fig_ztrades.add_hline(y=-z_stop, line_dash="dot", line_color="red")

        for t in trades:
            entry_i = t["Entry_index"]
            exit_i = t["Exit_index"]
            side = t["Side"]

            entry_color = "#00ff00" if "Long" in side else "#ff3300"
            exit_color = "white" if not t["Stopped"] else "#ff6600"

            fig_ztrades.add_scatter(
                x=[merged["datetime"].iloc[entry_i]],
                y=[zscore_bt.iloc[entry_i]],
                mode="markers",
                marker=dict(size=10, color=entry_color),
                showlegend=False
            )

            fig_ztrades.add_scatter(
                x=[merged["datetime"].iloc[exit_i]],
                y=[zscore_bt.iloc[exit_i]],
                mode="markers",
                marker=dict(size=10, color=exit_color, symbol="circle-open"),
                showlegend=False
            )

        fig_ztrades.update_layout(
            height=350,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Z-score",
        )

        st.plotly_chart(fig_ztrades, use_container_width=True)

        # ============================================================
        # Walk-Forward Hedge Ratio (β WF) - clean view
        # ============================================================

        st.markdown("### Walk-Forward Hedge Ratio (β WF)")

        fig_wf = go.Figure()

        # 1) Courbe β_WF
        fig_wf.add_scatter(
            x=merged["datetime"],
            y=beta_wf,
            mode="lines",
            name="β_WF",
            line=dict(color="#ffb347", width=2),
        )

        # 2) Traits verticaux aux changements de β
        beta_series = pd.Series(beta_wf)

        change_idx = beta_series[
            beta_series.notna() & (beta_series != beta_series.shift())
            ].index

        for i in change_idx:
            fig_wf.add_vline(
                x=merged["datetime"].iloc[i],
                line_width=1,
                line_color="rgba(0, 200, 255, 0.5)",
            )

        fig_wf.update_layout(
            height=220,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="β (WF)",
            showlegend=True,
        )

        st.plotly_chart(fig_wf, use_container_width=True)

        # ============================================================
        # WALK-FORWARD SEGMENT REPORT
        # ============================================================

        st.markdown("## Walk-Forward Segment Report")

        df_wf_report = compute_segment_metrics(merged, trades, beta_wf, segments)

        # Supprimer les segments sans trade
        df_wf_report = df_wf_report[df_wf_report["Trades"] > 0].reset_index(drop=True)

        if df_wf_report.empty:
            st.info("No walk-forward segments with trades for this configuration.")
        else:
            st.dataframe(df_wf_report.style.format({
                "Beta": "{:.3f}",
                "Winrate": "{:.1%}",
                "AvgPnL": "{:.4f}",
                "Sharpe": "{:.2f}",
                "MaxDD": "{:.1%}"
            }))

        # ============================================================
        # WALK-FORWARD HEATMAP
        # ============================================================

        st.markdown("### WF Segment Performance Heatmap")

        if df_wf_report.empty:
            st.info("Heatmap unavailable: no segments with trades.")
        else:
            heatmap_df = df_wf_report.copy()
            heatmap_df["Segment"] = heatmap_df["Segment"].astype(str)

            fig_heat = px.imshow(
                heatmap_df[["AvgPnL", "Winrate", "Sharpe"]].T,
                labels=dict(x="Segment", y="Metric", color="Value"),
                x=heatmap_df["Segment"],
                y=["AvgPnL", "Winrate", "Sharpe"],
                color_continuous_scale="RdYlGn"
            )

            fig_heat.update_layout(
                height=400,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20)
            )

            st.plotly_chart(fig_heat, use_container_width=True)


# ============================================================
# TAB 4 : SYNTHETIC TRAJECTORIES
# ============================================================

# ============================================================
# TAB 4 : SYNTHETIC TRAJECTORIES
# ============================================================

with tab_synth:
    st.subheader("Visualiser les trajectoires synthétiques")

    st.markdown("## Paramètres de génération")

    c1, c2 = st.columns(2)
    n_paths = c1.slider("Nombre de trajectoires", 5, 200, 20, 5)
    n_steps = c2.slider("Nombre de pas", 50, 2000, 500, 50)

    model = st.selectbox("Modèle", ["GBM", "OU", "Cointegrated PRO"])

    st.markdown("---")

    # Pré-init
    paths = None
    A = B = S = None

    # =============== GBM ===============
    if model == "GBM":
        c3, c4, c5 = st.columns(3)
        mu = c3.number_input("Drift μ", value=0.0002, format="%.6f")
        sigma = c4.number_input("Volatilité σ", value=0.01, format="%.6f")
        s0 = c5.number_input("Prix initial S₀", value=100.0)

    # =============== OU simple ===============
    elif model == "OU":
        st.markdown("### Paramètres OU estimés depuis la paire sélectionnée")

        mu_ou = st.session_state.get("ou_mu", 0.0)
        theta_ou = st.session_state.get("ou_theta", 0.0)
        sigma_ou = st.session_state.get("ou_sigma", 0.0)
        s0_ou = st.session_state.get("ou_s0", 0.0)

        c3, c4, c5, c6 = st.columns(4)
        mu_ou = c3.number_input("μ (mean reversion)", value=float(mu_ou), format="%.6f")
        theta_ou = c4.number_input("θ (speed)", value=float(theta_ou), format="%.6f")
        sigma_ou = c5.number_input("σ (vol résiduelle)", value=float(sigma_ou), format="%.6f")
        s0_ou = c6.number_input("S₀ (spread initial)", value=float(s0_ou), format="%.6f")

    # =============== Cointegrated PRO ===============
    else:
        st.markdown("### Paramètres calibrés automatiquement (Heston + OU)")

        params = calibrate_params_from_pair(df1, df2, spread, beta)

        c3, c4, c5 = st.columns(3)
        mu_s = c3.number_input("OU mean (μ_s)", value=float(params["mu_s"]), format="%.6f")
        theta_s = c4.number_input("OU speed (θ_s)", value=float(params["theta_s"]), format="%.6f")
        sigma_s = c5.number_input("OU vol (σ_s)", value=float(params["sigma_s"]), format="%.6f")

        c6, c7, c8 = st.columns(3)
        mu_m = c6.number_input("Market drift μ", value=float(params["mu"]), format="%.6f")
        v0 = c7.number_input("Initial variance v0", value=float(params["v0"]), format="%.6f")
        rho = c8.number_input("Corr price-vol ρ", value=float(params["rho"]), format="%.3f")

        beta_val = st.number_input("Hedge ratio β", value=float(params["beta"]), format="%.6f")

    st.markdown("---")

    run_syn = st.button("Générer")

    if run_syn:
        with st.spinner("Génération des trajectoires..."):

            if model == "GBM":
                paths = generate_synthetic_paths(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    mu=mu,
                    sigma=sigma,
                    s0=s0,
                )

            elif model == "OU":
                paths = generate_ou_paths(
                    n_paths=n_paths,
                    n_steps=n_steps,
                    mu=mu_ou,
                    theta=theta_ou,
                    sigma=sigma_ou,
                    s0=s0_ou,
                )

            else:  # Cointegrated PRO
                # on génère UNE paire A/B, pas n_paths (pour visualisation)
                A, B, X, S, v = simulate_cointegrated_assets(
                    n_steps,
                    beta_val,
                    X0=params["X0"], v0=v0, mu=mu_m,
                    theta=params["theta"], kappa=params["kappa"],
                    xi=params["xi"], rho=rho,
                    mu_s=mu_s, theta_s=theta_s,
                    sigma_s=sigma_s, S0=params["S0"],
                )

            st.markdown("### Trajectoires simulées")

            if model == "Cointegrated PRO":
                fig = go.Figure()
                fig.add_scatter(x=np.arange(n_steps), y=A, name="Asset A", mode="lines", line=dict(width=2))
                fig.add_scatter(x=np.arange(n_steps), y=B, name="Asset B", mode="lines", line=dict(width=2))
                fig.add_scatter(x=np.arange(n_steps), y=S, name="Spread (OU)", mode="lines", line=dict(width=1))
            else:
                fig = go.Figure()
                for i in range(min(n_paths, paths.shape[0])):
                    fig.add_scatter(
                        x=np.arange(n_steps),
                        y=paths[i],
                        mode="lines",
                        line=dict(width=1),
                        showlegend=False,
                    )

            fig.update_layout(
                height=400,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)


with tab_opt:
    st.subheader("Optimisation robuste (réel + synthétique)")

    c1, c2, c3 = st.columns(3)
    n_synth = c1.number_input("Nb trajectoires synthétiques", 10, 500, 50)
    n_param = c2.number_input("Nb de combinaisons de paramètres", 5, 200, 20)
    seed = c3.number_input("Seed RNG", 0, 999999, 42)

    st.markdown("### Plages de paramètres")

    p1, p2, p3, p4 = st.columns(4)
    z_entry_range = p1.slider("Z-entry", 1.0, 4.0, (1.2, 2.5), 0.1)
    z_exit_range = p2.slider("Z-exit", 0.0, 2.0, (0.2, 0.8), 0.05)
    z_window_range = p3.slider("Z-window", 20, 200, (40, 80), 5)
    wf_train_range = p4.slider("WF-train", 50, 500, (100, 200), 10)

    run_opt = st.button("Lancer optimisation")

    if run_opt:
        with st.spinner("Optimisation en cours..."):
            df_opt = run_full_optimization(
                merged,
                asset1,
                asset2,
                df1,
                df2,
                spread,
                beta,
                n_synth,
                n_param,
                z_entry_range,
                z_exit_range,
                z_window_range,
                wf_train_range,
                seed,
            )

            st.success("Optimisation terminée !")

            st.markdown("### Résultats")
            st.dataframe(df_opt, use_container_width=True)

            # Zone d'interaction
            st.markdown("### Explorer une configuration")

            selected_index = st.number_input(
                "Index de la ligne à explorer",
                min_value=0,
                max_value=len(df_opt) - 1,
                step=1,
                value=0
            )

            if st.button("Explorer cette configuration"):
                st.session_state["selected_opt_params"] = df_opt.iloc[selected_index].to_dict()
                st.experimental_rerun()

            if "selected_opt_params" in st.session_state:
                st.markdown("---")
                st.markdown("## 🔎 Exploration de la configuration sélectionnée")

                params = st.session_state["selected_opt_params"]

                # Affichage des paramètres
                st.json(params)

                # Recalcul du backtest réel
                spread_bt, zscore_bt, beta_wf = walk_forward_beta_spread_zscore(
                    merged[f"norm_{asset1}"],
                    merged[f"norm_{asset2}"],
                    train=int(params["wf_train"]),
                    test=int(params["wf_test"]),
                    z_window=int(params["z_window"]),
                )

                equity, trades, pnl_list = backtest_pair(
                    spread_bt,
                    zscore_bt,
                    merged[f"norm_{asset1}"],
                    merged[f"norm_{asset2}"],
                    beta,
                    z_entry=float(params["z_entry"]),
                    z_exit=float(params["z_exit"]),
                    z_stop=float(params["z_entry"]) * 2,
                    fees=0.0002,
                )

                # Figures
                st.markdown("### 📈 Equity Curve")
                fig_eq = go.Figure()
                fig_eq.add_scatter(x=merged["datetime"], y=equity, mode="lines")
                st.plotly_chart(fig_eq, use_container_width=True)

                # Résumé métriques
                st.markdown("### 📊 Metrics")
                st.write({
                    "Sharpe_real": params["Sharpe_real"],
                    "Trades": len(trades),
                    "Max DD": float(np.min(equity) / np.max(equity) - 1) if len(equity) > 1 else 0
                })

                # Liste trades
                st.markdown("### 📋 Trade List")
                if len(trades) > 0:
                    st.dataframe(pd.DataFrame(trades), use_container_width=True)
                else:
                    st.info("Aucun trade pour cette configuration.")







