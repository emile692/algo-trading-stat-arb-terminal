# backtesting/global_loop.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from object.class_file import BatchConfig, StrategyParams
from backtesting.functions import precompute_pair_state_for_window
from backtesting.engine import run_daily_portfolio_engine

from utils.loader import load_price_csv


def _pca_mode_strength_and_mkt_vol_asof(
    rets: pd.DataFrame,
    eval_dates: pd.DatetimeIndex,
    pca_window: int,
    min_assets: int,
    mkt_vol_window: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calcule (sans lookahead) sur chaque date t de eval_dates:
      - mode_strength[t] = lambda1 / N (PCA sur corrélation des returns), fenêtre finissant à t-1
      - mkt_vol[t] = std(rolling) du return de marché equal-weight, fenêtre finissant à t-1

    rets: DataFrame returns (index datetime, columns assets)
    """
    rets = rets.sort_index()
    out_mode = {}
    out_vol = {}

    for t in eval_dates:
        t = pd.to_datetime(t).normalize()
        end_dt = (t - BDay(1)).normalize()

        # Fenêtre PCA (corr)
        w = rets.loc[:end_dt].tail(pca_window)
        if w.shape[0] < pca_window:
            out_mode[t] = np.nan
        else:
            w = w.dropna(axis=1, how="any")
            if w.shape[1] < min_assets:
                out_mode[t] = np.nan
            else:
                X = w.to_numpy(dtype=float)

                # standardize -> corr
                X -= X.mean(axis=0, keepdims=True)
                std = X.std(axis=0, ddof=1, keepdims=True)
                ok = (std.reshape(-1) > 0) & np.isfinite(std.reshape(-1))
                if ok.sum() < min_assets:
                    out_mode[t] = np.nan
                else:
                    X = X[:, ok]
                    X = X / std[:, ok]

                    T, N = X.shape
                    C = (X.T @ X) / float(T - 1)
                    evals = np.linalg.eigvalsh(C)  # ascending
                    lam1 = float(evals[-1])
                    out_mode[t] = lam1 / float(N)

        # Vol marché equal-weight (sur toutes colonnes dispo ce jour)
        wv = rets.loc[:end_dt].tail(mkt_vol_window)
        if wv.shape[0] < mkt_vol_window:
            out_vol[t] = np.nan
        else:
            mkt = wv.mean(axis=1)  # equal-weight market return proxy
            out_vol[t] = float(mkt.std(ddof=1)) if mkt.notna().sum() >= max(10, mkt_vol_window // 2) else np.nan

    s_mode = pd.Series(out_mode).sort_index()
    s_mode.name = "pca_mode_strength"

    s_vol = pd.Series(out_vol).sort_index()
    s_vol.name = "mkt_vol"

    return s_mode, s_vol


@lru_cache(maxsize=512)
def _load_asset_prices(asset: str, data_path_str: str) -> pd.Series:
    """
    Retourne une Series de close indexée par datetime.
    Cache pour éviter de relire le CSV à chaque appel.
    """
    df = load_price_csv(asset.upper(), Path(data_path_str)).copy()
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.normalize()
    s = df.set_index("datetime")["close"].astype(float).sort_index()
    s.name = asset.upper()
    return s


def build_price_panel(cfg: BatchConfig, assets: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    series = []
    for a in sorted(set(assets)):
        try:
            series.append(_load_asset_prices(a, str(cfg.data_path)))
        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    prices = pd.concat(series, axis=1).sort_index()

    # buffer pour PCA + vol + exec lag
    # (si tu augmentes PCA_WINDOW, augmente ce buffer)
    prices = prices.loc[start - BDay(400): end]
    return prices


def _load_scans(cfg: BatchConfig, universes: List[str], scans: Optional[pd.DataFrame]) -> pd.DataFrame:
    if scans is not None:
        df = scans.copy()
        df["scan_date"] = pd.to_datetime(df["scan_date"]).dt.normalize()
        if "universe" not in df.columns:
            df["universe"] = "INLINE"
        return df

    if cfg.scanner_path is None:
        raise ValueError("scanner_path must be set if scans is None.")

    scanner_dir = Path(cfg.scanner_path)
    frames = []
    for universe in universes:
        fp = scanner_dir / f"{universe}.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp)
        if d.empty:
            continue
        d["scan_date"] = pd.to_datetime(d["scan_date"]).dt.normalize()
        d["universe"] = universe
        frames.append(d)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def run_global_ranking_daily_portfolio(
    cfg: BatchConfig,
    params: StrategyParams,
    universes: List[str],
    scans: Optional[pd.DataFrame] = None,
) -> Dict:

    scans_df = _load_scans(cfg, universes, scans)
    if scans_df.empty:
        return {}

    start = pd.to_datetime(cfg.start_date).normalize()
    end = pd.to_datetime(cfg.end_date).normalize()

    # -------------------- ASOF scan settings --------------------
    MAX_SCAN_AGE_BDAYS = 15  # weekly ~ 7-10, biweekly ~ 12-15, None => no cutoff

    extra = params.exec_lag_days + 5
    if MAX_SCAN_AGE_BDAYS is not None:
        extra += MAX_SCAN_AGE_BDAYS

    scans_df = scans_df[
        (scans_df["scan_date"] >= start - BDay(max(5, extra)))
        & (scans_df["scan_date"] <= end)
    ].copy()
    if scans_df.empty:
        return {}

    scans_by_date = {d: g for d, g in scans_df.groupby("scan_date", sort=True)}
    scan_dates_idx = pd.DatetimeIndex(sorted(scans_by_date.keys())).sort_values()

    def _scan_date_asof(target: pd.Timestamp) -> Optional[pd.Timestamp]:
        target = pd.to_datetime(target).normalize()
        sd = scan_dates_idx.asof(target)  # last <= target
        if pd.isna(sd):
            return None
        return pd.to_datetime(sd).normalize()

    # assets nécessaires = ceux apparaissant dans les scans (buffer inclus)
    all_assets = (
        pd.concat([scans_df["asset_1"], scans_df["asset_2"]], ignore_index=True)
          .astype(str).str.upper().unique().tolist()
    )

    price_df = build_price_panel(cfg, all_assets, start, end)
    if price_df.empty:
        return {}

    # -------------------- Regime filter (PCA mode + market vol) --------------------
    # Avec N~18, mode_strength ~ corr moyenne. Pour rendre le filtre utile:
    # - fenêtre plus longue (stabilité)
    # - gate conjonctif avec vol (coupe surtout pendant phases de stress)
    PCA_WINDOW = 252            # 120/252 recommandé
    PCA_Q = 0.85                # viser 10-25% de jours "bloqués" plutôt que 5%
    PCA_MIN_ASSETS = 12         # avec 18 titres, 12+ est un bon compromis

    MKT_VOL_WINDOW = 20
    MKT_VOL_Q = 0.80            # ex: 0.75/0.80/0.85

    # calcul seulement sur dates de scan (weekly/biweekly friendly), ffill ensuite
    trade_dates = pd.bdate_range(start=start, end=end).normalize()
    eval_dates = scan_dates_idx[(scan_dates_idx >= start) & (scan_dates_idx <= end)]
    if len(eval_dates) == 0:
        return {}

    rets = price_df.pct_change(fill_method=None)

    pca_mode_scan, mkt_vol_scan = _pca_mode_strength_and_mkt_vol_asof(
        rets=rets,
        eval_dates=eval_dates,
        pca_window=PCA_WINDOW,
        min_assets=PCA_MIN_ASSETS,
        mkt_vol_window=MKT_VOL_WINDOW,
    )

    pca_mode = pca_mode_scan.reindex(trade_dates).ffill()
    mkt_vol = mkt_vol_scan.reindex(trade_dates).ffill()

    # seuils "online" (pas de lookahead) via expanding quantile
    # min_periods: au moins ~1 an de bourse avant d'activer vraiment le filtre
    pca_thresh_series = (
        pca_mode.expanding(min_periods=252).quantile(PCA_Q)
        if not pca_mode.dropna().empty else None
    )
    vol_thresh_series = (
        mkt_vol.expanding(min_periods=252).quantile(MKT_VOL_Q)
        if not mkt_vol.dropna().empty else None
    )

    def get_ranked_pairs(dt: pd.Timestamp) -> List[Tuple[str, str]]:
        dt = pd.to_datetime(dt).normalize()

        # ----- Regime gate (conjonctif) -----
        mode_val = pca_mode.loc[dt] if dt in pca_mode.index else np.nan
        vol_val = mkt_vol.loc[dt] if dt in mkt_vol.index else np.nan

        mode_th = pca_thresh_series.loc[dt] if pca_thresh_series is not None and dt in pca_thresh_series.index else np.inf
        vol_th = vol_thresh_series.loc[dt] if vol_thresh_series is not None and dt in vol_thresh_series.index else np.inf

        # Gate: on bloque seulement si "market mode" élevé ET vol élevée.
        if pd.isna(mode_val) or pd.isna(vol_val) or pd.isna(mode_th) or pd.isna(vol_th):
            # si pas assez d'historique, on laisse passer (sinon tu bloques au début)
            pass
        else:
            if (mode_val >= mode_th) and (vol_val >= vol_th):
                return []

        # ----- asof scan lookup -----
        scan_target = (dt - BDay(params.exec_lag_days)).normalize()
        scan_dt = _scan_date_asof(scan_target)
        if scan_dt is None:
            return []

        if MAX_SCAN_AGE_BDAYS is not None:
            if scan_dt < (scan_target - BDay(MAX_SCAN_AGE_BDAYS)):
                return []

        df_day = scans_by_date.get(scan_dt)
        if df_day is None or df_day.empty:
            return []

        eligible = df_day[df_day["eligibility"] == "ELIGIBLE"]
        if eligible.empty:
            return []

        ranked = (
            eligible.sort_values("eligibility_score", ascending=False)
                    .head(params.top_n_candidates)
        )

        return list(
            zip(
                ranked["asset_1"].astype(str).str.upper(),
                ranked["asset_2"].astype(str).str.upper(),
            )
        )

    def get_pair_state(dt: pd.Timestamp, pairs: List[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
        cand_df = pd.DataFrame([{"asset_1": a1, "asset_2": a2} for (a1, a2) in pairs])
        return precompute_pair_state_for_window(
            cfg=cfg,
            params=params,
            candidates=cand_df,
            start=dt,
            end=dt,
        )

    res = run_daily_portfolio_engine(
        params=params,
        start=start,
        end=end,
        get_ranked_pairs=get_ranked_pairs,
        get_pair_state=get_pair_state,
    )

    if not res:
        return {}

    equity = res["equity"].copy()
    trades = res["trades"].copy()

    # reporting / stats
    equity["trade_month"] = pd.to_datetime(equity["datetime"]).dt.strftime("%Y-%m")

    monthly = (
        equity.groupby("trade_month", as_index=False)
              .agg(
                  start_equity=("equity", "first"),
                  end_equity=("equity", "last"),
                  n_days=("equity", "size"),
                  max_open_positions=("n_open_positions", "max"),
              )
    )
    monthly["month_return"] = monthly["end_equity"] / monthly["start_equity"] - 1.0

    returns = equity["equity"].pct_change().dropna()
    final_eq = float(equity["equity"].iloc[-1])

    n = len(equity)
    cagr = (final_eq ** (252 / n) - 1.0) if n > 0 else np.nan
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(np.sqrt(252) * returns.mean() / vol) if (vol is not None and vol > 0) else np.nan
    mdd = float((equity["equity"] / equity["equity"].cummax() - 1).min())

    # stats regime debug
    blocked = 0
    denom = 0
    for dt in trade_dates:
        mode_val = pca_mode.loc[dt] if dt in pca_mode.index else np.nan
        vol_val = mkt_vol.loc[dt] if dt in mkt_vol.index else np.nan
        mode_th = pca_thresh_series.loc[dt] if pca_thresh_series is not None and dt in pca_thresh_series.index else np.nan
        vol_th = vol_thresh_series.loc[dt] if vol_thresh_series is not None and dt in vol_thresh_series.index else np.nan
        if np.isfinite(mode_val) and np.isfinite(vol_val) and np.isfinite(mode_th) and np.isfinite(vol_th):
            denom += 1
            if (mode_val >= mode_th) and (vol_val >= vol_th):
                blocked += 1
    pct_blocked = (blocked / denom) if denom > 0 else np.nan

    stats = {
        "Final Equity": round(final_eq, 2),
        "CAGR": round(float(cagr), 3) if not np.isnan(cagr) else np.nan,
        "Sharpe": round(float(sharpe), 2) if not np.isnan(sharpe) else np.nan,
        "Max Drawdown": round(mdd, 3),
        "Nb Trades": int(len(trades)) if isinstance(trades, pd.DataFrame) else 0,

        "PCA window": PCA_WINDOW,
        "PCA q": PCA_Q,
        "PCA min assets": PCA_MIN_ASSETS,
        "Mkt vol window": MKT_VOL_WINDOW,
        "Mkt vol q": MKT_VOL_Q,
        "% days blocked (when active)": round(float(pct_blocked), 3) if np.isfinite(pct_blocked) else np.nan,
        "Max scan age (bdays)": MAX_SCAN_AGE_BDAYS,
    }

    return {"equity": equity, "monthly": monthly, "trades": trades, "stats": stats}