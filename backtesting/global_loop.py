# backtesting/global_loop.py

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from object.class_file import BatchConfig, StrategyParams
from backtesting.engine import run_daily_portfolio_engine

from utils.loader import load_price_csv


MAX_SCAN_AGE_BDAYS = 15

PCA_WINDOW = 252
PCA_Q = 0.90
PCA_MIN_ASSETS = 12

MKT_VOL_WINDOW = 20
MKT_VOL_Q = 0.80

_NS_PER_DAY = np.int64(86_400_000_000_000)
_CONTEXT_CACHE_MAXSIZE = 8
_PAIR_STATE_CACHE_MAXSIZE = 24

_CACHE_LOCK = RLock()
_GLOBAL_CONTEXT_CACHE: OrderedDict[Tuple, "_GlobalRunContext"] = OrderedDict()
_PAIR_STATE_CACHE: OrderedDict[Tuple, Dict[str, pd.DataFrame]] = OrderedDict()
_SCAN_SIG_MEMO: Dict[int, Tuple[int, Tuple[str, ...], Tuple]] = {}
_VALID_SCAN_WEEKDAYS = {"MON", "TUE", "WED", "THU", "FRI"}
_LEGACY_SELECTION_SCORE_VARIANTS = {
    "baseline",
    "half_life_weighted",
    "spread_speed_penalized",
    "distance_to_mean_over_half_life",
    "low_corr_penalized",
}
_COMPOSITE_SELECTION_SCORE_VARIANTS = {
    "baseline",
    "rank_percentile",
    "robust_zscore",
    "rank_stability_penalty",
}


@dataclass
class _GlobalRunContext:
    key: Tuple
    start: pd.Timestamp
    end: pd.Timestamp
    trade_dates: pd.DatetimeIndex
    ranked_pairs_by_date: Dict[pd.Timestamp, List[Tuple[str, str]]]
    scan_date_by_trade_date: Dict[pd.Timestamp, pd.Timestamp]
    all_ranked_pairs: List[Tuple[str, str]]
    signal_price_panel: pd.DataFrame
    asset_log_offsets: Dict[str, float]
    pct_blocked: float
    signal_space: str


class GlobalRankingSweepRunner:
    """
    Lightweight runner for parameter sweeps.
    Reuses global precomputations and memoizes final Sharpe per (z_window, z_entry).
    """

    def __init__(
        self,
        cfg: BatchConfig,
        scans: pd.DataFrame,
        universes: List[str],
        *,
        fees: float,
        top_n_candidates: int,
        max_positions: int,
        max_holding_days: int,
        beta_mode: str = "static",
        signal_space: str = "raw",
        pca_signal_window: int = 252,
        pca_signal_components: int = 3,
        pca_signal_min_assets: int = 10,
    ) -> None:
        self.cfg = cfg
        self.scans = scans
        self.universes = list(universes)
        self._params_template = StrategyParams(
            z_entry=2.0,
            z_exit=0.5,
            z_stop=4.0,
            z_window=60,
            beta_mode=beta_mode,
            fees=float(fees),
            top_n_candidates=int(top_n_candidates),
            max_positions=int(max_positions),
            max_holding_days=int(max_holding_days),
            signal_space=signal_space,
            pca_signal_window=int(pca_signal_window),
            pca_signal_components=int(pca_signal_components),
            pca_signal_min_assets=int(pca_signal_min_assets),
        )
        self._result_cache: Dict[Tuple[int, float], float] = {}

    def run_for_params(self, z_window: int, z_entry: float) -> float:
        key = (int(z_window), float(z_entry))
        cached = self._result_cache.get(key)
        if cached is not None:
            return cached

        params = replace(
            self._params_template,
            z_entry=float(z_entry),
            z_exit=float(z_entry) / 3.0,
            z_stop=2.0 * float(z_entry),
            z_window=int(z_window),
        )
        res = run_global_ranking_daily_portfolio(
            cfg=self.cfg,
            params=params,
            universes=self.universes,
            scans=self.scans,
        )
        sharpe = float("nan") if not res else float(res["stats"]["Sharpe"])
        self._result_cache[key] = sharpe
        return sharpe


def run_for_params(
    z_window: int,
    z_entry: float,
    cfg: BatchConfig,
    scans: pd.DataFrame,
    universes: List[str],
    *,
    fees: float,
    top_n_candidates: int,
    max_positions: int,
    max_holding_days: int,
    beta_mode: str = "static",
    signal_space: str = "raw",
    pca_signal_window: int = 252,
    pca_signal_components: int = 3,
    pca_signal_min_assets: int = 10,
) -> float:
    """
    Stateless helper mirroring notebook-style run_for_params.
    Prefer GlobalRankingSweepRunner for large sweeps to maximize cache reuse.
    """
    params = StrategyParams(
        z_entry=float(z_entry),
        z_exit=float(z_entry) / 3.0,
        z_stop=2.0 * float(z_entry),
        z_window=int(z_window),
        beta_mode=beta_mode,
        fees=float(fees),
        top_n_candidates=int(top_n_candidates),
        max_positions=int(max_positions),
        max_holding_days=int(max_holding_days),
        signal_space=signal_space,
        pca_signal_window=int(pca_signal_window),
        pca_signal_components=int(pca_signal_components),
        pca_signal_min_assets=int(pca_signal_min_assets),
    )
    res = run_global_ranking_daily_portfolio(
        cfg=cfg,
        params=params,
        universes=universes,
        scans=scans,
    )
    if not res:
        return float("nan")
    return float(res["stats"]["Sharpe"])


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


def _rolling_pca_idio_returns(
    returns: pd.DataFrame,
    window: int,
    n_components: int,
    min_assets: int,
) -> pd.DataFrame:
    """
    Build idiosyncratic returns r_idio(t) from rolling PCA de-factorization.
    For each date t:
      - PCA is fit on [t-window, t-1] only (no lookahead),
      - each asset is regressed on PC1..PCk over that window,
      - residual at t is stored.
    """
    rets = returns.sort_index().astype(float)
    out = pd.DataFrame(np.nan, index=rets.index, columns=rets.columns, dtype=float)

    min_req = max(int(min_assets), int(n_components) + 2)
    idx = rets.index

    for i in range(int(window), len(idx)):
        dt = idx[i]
        hist = rets.iloc[i - int(window):i]

        # Keep assets complete over the train window and observed on t.
        valid = hist.notna().all(axis=0) & rets.iloc[i].notna()
        assets = hist.columns[valid.values]
        if len(assets) < min_req:
            continue

        X = hist[assets].to_numpy(dtype=float)        # T x N
        r_t = rets.loc[dt, assets].to_numpy(dtype=float)  # N

        mu = X.mean(axis=0, keepdims=True)
        sd = X.std(axis=0, ddof=1, keepdims=True)
        good = np.isfinite(sd).reshape(-1) & (sd.reshape(-1) > 0.0)
        if int(good.sum()) < min_req:
            continue

        assets = np.asarray(assets)[good]
        X = X[:, good]
        r_t = r_t[good]
        mu = mu[:, good]
        sd = sd[:, good]

        X_std = (X - mu) / sd
        C = (X_std.T @ X_std) / float(X_std.shape[0] - 1)

        _, evecs = np.linalg.eigh(C)
        # Use at most N-1 components to keep an idiosyncratic remainder.
        k = int(min(int(n_components), max(1, evecs.shape[1] - 1)))
        if k < 1:
            continue

        V = evecs[:, -k:]                # N x k
        F = X_std @ V                    # T x k

        G = np.hstack([np.ones((X.shape[0], 1)), F])
        coef = np.linalg.pinv(G.T @ G) @ (G.T @ X)  # (1+k) x N

        x_t_std = (r_t - mu.reshape(-1)) / sd.reshape(-1)
        f_t = x_t_std @ V
        g_t = np.concatenate([[1.0], f_t])
        r_hat_t = g_t @ coef

        out.loc[dt, assets] = r_t - r_hat_t

    return out


def _build_idio_price_panel_from_raw_prices(
    price_df: pd.DataFrame,
    pca_window: int,
    n_components: int,
    min_assets: int,
) -> pd.DataFrame:
    """
    Convert rolling PCA idiosyncratic returns into a positive synthetic price panel.
    """
    rets = price_df.pct_change(fill_method=None)
    idio_rets = _rolling_pca_idio_returns(
        returns=rets,
        window=pca_window,
        n_components=n_components,
        min_assets=min_assets,
    )

    # Guard log1p against rare <= -1 residuals.
    safe = idio_rets.clip(lower=-0.999999)
    idio_prices = np.exp(np.log1p(safe.fillna(0.0)).cumsum())
    idio_prices = idio_prices.where(price_df.notna())
    return idio_prices


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


def build_price_panel(
    cfg: BatchConfig,
    assets: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    buffer_bdays: int = 400,
) -> pd.DataFrame:
    series = []
    for a in sorted(set(assets)):
        try:
            series.append(_load_asset_prices(a, str(cfg.data_path)))
        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    prices = pd.concat(series, axis=1).sort_index()

    # Buffer for PCA/vol/signal warmups.
    prices = prices.loc[start - BDay(int(buffer_bdays)): end]
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


def _lru_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _lru_put(cache: OrderedDict, key, value, maxsize: int) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > int(maxsize):
        cache.popitem(last=False)


def _scanner_files_signature(cfg: BatchConfig, universes: List[str]) -> Tuple:
    if cfg.scanner_path is None:
        return ("scanner", None)
    scanner_dir = Path(cfg.scanner_path)
    entries = []
    for u in sorted(set(universes)):
        fp = scanner_dir / f"{u}.parquet"
        if not fp.exists():
            entries.append((u, False, None, None))
            continue
        st = fp.stat()
        entries.append((u, True, int(st.st_size), int(st.st_mtime_ns)))
    return ("scanner", str(scanner_dir.resolve()), tuple(entries))


def _scan_frame_signature(scans: Optional[pd.DataFrame]) -> Tuple:
    if scans is None:
        return ("scan_df", None)

    obj_id = id(scans)
    cols = tuple(str(c) for c in scans.columns)
    memo = _SCAN_SIG_MEMO.get(obj_id)
    if memo is not None and memo[0] == len(scans) and memo[1] == cols:
        return memo[2]

    if scans.empty:
        sig = ("scan_df", 0, cols, 0)
        _SCAN_SIG_MEMO[obj_id] = (0, cols, sig)
        return sig

    keep_cols = [
        c for c in ("scan_date", "asset_1", "asset_2", "eligibility", "eligibility_score", "universe")
        if c in scans.columns
    ]
    if not keep_cols:
        sig = ("scan_df", len(scans), cols, 0)
        _SCAN_SIG_MEMO[obj_id] = (len(scans), cols, sig)
        return sig

    hdf = scans[keep_cols].copy()
    if "scan_date" in hdf.columns:
        hdf["scan_date"] = pd.to_datetime(hdf["scan_date"]).dt.normalize()
    for c in ("asset_1", "asset_2", "eligibility", "universe"):
        if c in hdf.columns:
            hdf[c] = hdf[c].astype(str).str.upper()

    hv = pd.util.hash_pandas_object(hdf, index=False, categorize=True).to_numpy(dtype=np.uint64)
    if hv.size == 0:
        hash64 = 0
    else:
        hash64 = int(np.bitwise_xor.reduce(hv))

    sig = ("scan_df", len(scans), cols, hash64)
    _SCAN_SIG_MEMO[obj_id] = (len(scans), cols, sig)
    return sig


def _normalize_eligibility_labels(labels: Tuple[str, ...]) -> Tuple[str, ...]:
    if not labels:
        return ("ELIGIBLE",)
    out = tuple(sorted({str(x).strip().upper() for x in labels if str(x).strip()}))
    return out if out else ("ELIGIBLE",)


def _normalize_scan_frequency(freq: str) -> str:
    f = str(freq).strip().lower()
    if f in {"", "daily", "day", "d", "b"}:
        return "daily"
    if f in {"weekly", "week", "w"}:
        return "weekly"
    raise ValueError(f"Unsupported scan_frequency='{freq}'. Use 'daily' or 'weekly'.")


def _normalize_scan_weekday(scan_weekday: str) -> str:
    wd = str(scan_weekday).strip().upper()
    if wd not in _VALID_SCAN_WEEKDAYS:
        raise ValueError(f"Unsupported scan_weekday='{scan_weekday}'. Use one of {sorted(_VALID_SCAN_WEEKDAYS)}.")
    return wd


def _apply_scan_schedule(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    if df.empty:
        return df

    freq = _normalize_scan_frequency(params.scan_frequency)
    if freq == "daily":
        return df

    wd = _normalize_scan_weekday(params.scan_weekday)
    out = df.copy()
    out["scan_date"] = pd.to_datetime(out["scan_date"]).dt.normalize()
    out = out[out["scan_date"].dt.dayofweek <= 4].copy()
    if out.empty:
        return out
    out["_scan_bucket"] = out["scan_date"].dt.to_period(f"W-{wd}")

    group_cols = ["_scan_bucket"]
    if "universe" in out.columns:
        group_cols.insert(0, "universe")

    out["_scan_pick"] = out.groupby(group_cols)["scan_date"].transform("max")
    out = out[out["scan_date"] == out["_scan_pick"]].copy()
    out = out.drop(columns=["_scan_bucket", "_scan_pick"])
    return out.reset_index(drop=True)


def _rank_pct_series(s: pd.Series, ascending: bool) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if int(x.notna().sum()) <= 1:
        return pd.Series(0.5, index=s.index, dtype=float)
    return x.rank(pct=True, ascending=ascending, method="average").astype(float)


def _selection_variant_for_mode(params: StrategyParams) -> str:
    mode = str(params.selection_mode).strip().lower()
    variant = str(params.selection_score_variant).strip().lower() or "baseline"

    if mode == "legacy":
        allowed = _LEGACY_SELECTION_SCORE_VARIANTS
    elif mode == "composite_quality":
        allowed = _COMPOSITE_SELECTION_SCORE_VARIANTS
    else:
        raise ValueError(f"Unsupported selection_mode='{params.selection_mode}'. Use 'legacy' or 'composite_quality'.")

    if variant not in allowed:
        raise ValueError(
            "Unsupported selection_score_variant="
            f"'{params.selection_score_variant}' for selection_mode='{mode}'. "
            f"Use one of {sorted(allowed)}."
        )
    return variant


def _positive_strength_or_default(value: float, fallback: float) -> float:
    try:
        val = float(value)
    except Exception:
        return float(fallback)
    if (not np.isfinite(val)) or val <= 0.0:
        return float(fallback)
    return float(val)


def _winsorize_series(s: pd.Series, q: float) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if not np.isfinite(q) or q <= 0.0:
        return x
    qq = float(min(0.25, max(0.0, q)))
    if int(x.notna().sum()) < 5:
        return x
    lo = float(x.quantile(qq))
    hi = float(x.quantile(1.0 - qq))
    return x.clip(lower=lo, upper=hi)


def _robust_z_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = float(x.median()) if int(x.notna().sum()) > 0 else np.nan
    mad = float((x - med).abs().median()) if np.isfinite(med) else np.nan
    if (not np.isfinite(mad)) or mad <= 1e-12:
        return pd.Series(0.0, index=s.index, dtype=float)
    z = 0.6745 * (x - med) / mad
    return pd.to_numeric(z, errors="coerce").fillna(0.0)


def _compute_selection_score(df: pd.DataFrame, params: StrategyParams) -> pd.Series:
    mode = str(params.selection_mode).strip().lower()
    variant = _selection_variant_for_mode(params)

    s_elig = pd.to_numeric(df.get("eligibility_score", pd.Series(index=df.index, dtype=float)), errors="coerce")
    s_corr = pd.to_numeric(df.get("12m_corr", pd.Series(index=df.index, dtype=float)), errors="coerce").abs()
    s_hl = pd.to_numeric(df.get("6m_half_life", pd.Series(index=df.index, dtype=float)), errors="coerce")
    s_abs_last_z = pd.to_numeric(df.get("6m_abs_last_z", pd.Series(index=df.index, dtype=float)), errors="coerce")
    s_speed = pd.to_numeric(df.get("6m_mean_abs_delta_z", pd.Series(index=df.index, dtype=float)), errors="coerce")
    if int(s_speed.notna().sum()) == 0:
        s_speed = pd.to_numeric(df.get("6m_mean_abs_delta_spread", pd.Series(index=df.index, dtype=float)), errors="coerce")
    if int(s_speed.notna().sum()) == 0:
        s_speed = pd.to_numeric(df.get("6m_spread_std", pd.Series(index=df.index, dtype=float)), errors="coerce")

    if mode == "legacy":
        base_score = s_elig
        if variant == "baseline":
            return base_score.fillna(-np.inf)

        if variant == "half_life_weighted":
            lam = _positive_strength_or_default(params.selection_half_life_penalty, 0.15)
            score = base_score - lam * _rank_pct_series(s_hl, ascending=True)
            return pd.to_numeric(score, errors="coerce").fillna(-np.inf)

        if variant == "spread_speed_penalized":
            lam = _positive_strength_or_default(params.selection_speed_penalty, 0.10)
            if int(s_speed.notna().sum()) == 0:
                return base_score.fillna(-np.inf)
            score = base_score - lam * _rank_pct_series(s_speed, ascending=True)
            return pd.to_numeric(score, errors="coerce").fillna(-np.inf)

        if variant == "distance_to_mean_over_half_life":
            lam = _positive_strength_or_default(params.selection_distance_weight, 0.15)
            hl_safe = s_hl.clip(lower=1.0)
            signal = (s_abs_last_z / np.sqrt(hl_safe)).replace([np.inf, -np.inf], np.nan)
            if int(signal.notna().sum()) == 0:
                return base_score.fillna(-np.inf)
            score = base_score + lam * _rank_pct_series(signal, ascending=True)
            return pd.to_numeric(score, errors="coerce").fillna(-np.inf)

        lam = _positive_strength_or_default(params.selection_corr_penalty, 0.10)
        score = base_score - lam * _rank_pct_series(s_corr, ascending=False)
        return pd.to_numeric(score, errors="coerce").fillna(-np.inf)

    if mode != "composite_quality":
        raise ValueError(f"Unsupported selection_mode='{params.selection_mode}'. Use 'legacy' or 'composite_quality'.")
    s_nvw = pd.to_numeric(df.get("n_valid_windows", pd.Series(index=df.index, dtype=float)), errors="coerce")
    s_beta = pd.to_numeric(df.get("beta_std", pd.Series(index=df.index, dtype=float)), errors="coerce")
    s_sstd = pd.to_numeric(df.get("6m_spread_std", pd.Series(index=df.index, dtype=float)), errors="coerce")

    base_score = (
        1.00 * _rank_pct_series(s_elig, ascending=True)
        + 0.90 * _rank_pct_series(s_corr, ascending=True)
        + 0.75 * _rank_pct_series(s_nvw, ascending=True)
        + 0.60 * _rank_pct_series(s_hl, ascending=False)
        + 0.40 * _rank_pct_series(s_beta, ascending=False)
        + 0.25 * _rank_pct_series(s_sstd, ascending=True)
    )

    if variant == "baseline":
        score = base_score
    elif variant == "rank_percentile":
        score = (
            _rank_pct_series(s_elig, ascending=True)
            + _rank_pct_series(s_corr, ascending=True)
            + _rank_pct_series(s_nvw, ascending=True)
            + _rank_pct_series(s_hl, ascending=False)
            + _rank_pct_series(s_beta, ascending=False)
            + _rank_pct_series(s_sstd, ascending=True)
        ) / 6.0
    elif variant == "robust_zscore":
        winsor_q = float(params.selection_winsor_quantile)
        if winsor_q <= 0.0:
            winsor_q = 0.05
        score = (
            1.00 * _robust_z_series(_winsorize_series(s_elig, winsor_q))
            + 0.90 * _robust_z_series(_winsorize_series(s_corr, winsor_q))
            + 0.75 * _robust_z_series(_winsorize_series(s_nvw, winsor_q))
            - 0.60 * _robust_z_series(_winsorize_series(s_hl, winsor_q))
            - 0.40 * _robust_z_series(_winsorize_series(s_beta, winsor_q))
            + 0.25 * _robust_z_series(_winsorize_series(s_sstd, winsor_q))
        )
    else:  # rank_stability_penalty
        lam = float(params.selection_stability_penalty)
        if lam <= 0.0:
            lam = 0.35
        stability_penalty = (
            0.50 * _rank_pct_series(s_beta, ascending=True)
            + 0.30 * _rank_pct_series(s_hl, ascending=True)
            + 0.20 * _rank_pct_series(s_nvw, ascending=False)
        )
        score = base_score - lam * stability_penalty

    return pd.to_numeric(score, errors="coerce").fillna(-np.inf)


def _apply_scan_filters(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    if df.empty:
        return df

    m = pd.Series(True, index=df.index, dtype=bool)
    if params.min_corr_12m is not None:
        corr = pd.to_numeric(df.get("12m_corr", pd.Series(index=df.index, dtype=float)), errors="coerce").abs()
        m &= corr >= float(params.min_corr_12m)
    if params.max_half_life_6m is not None:
        hl = pd.to_numeric(df.get("6m_half_life", pd.Series(index=df.index, dtype=float)), errors="coerce")
        m &= hl <= float(params.max_half_life_6m)
    if params.max_beta_std is not None:
        bs = pd.to_numeric(df.get("beta_std", pd.Series(index=df.index, dtype=float)), errors="coerce")
        m &= bs <= float(params.max_beta_std)
    if params.min_spread_std_6m is not None:
        sstd = pd.to_numeric(df.get("6m_spread_std", pd.Series(index=df.index, dtype=float)), errors="coerce")
        m &= sstd >= float(params.min_spread_std_6m)
    if params.min_n_valid_windows is not None:
        nvw = pd.to_numeric(df.get("n_valid_windows", pd.Series(index=df.index, dtype=float)), errors="coerce")
        m &= nvw >= int(params.min_n_valid_windows)

    return df[m].copy()


def _select_ranked_pairs_for_scan_day(
    df_day: pd.DataFrame,
    params: StrategyParams,
    top_n: int,
) -> List[Tuple[str, str]]:
    if df_day.empty or top_n <= 0:
        return []

    labels = _normalize_eligibility_labels(tuple(params.eligibility_labels))
    pool = df_day[df_day["eligibility"].astype(str).str.upper().isin(labels)].copy()
    if pool.empty:
        return []

    pool = _apply_scan_filters(pool, params)
    if pool.empty:
        return []

    pool["_selection_score"] = _compute_selection_score(pool, params)
    pool = pool.sort_values("_selection_score", ascending=False)

    max_pairs_per_asset = int(params.max_pairs_per_asset)
    if max_pairs_per_asset <= 0:
        picked = pool.head(top_n)
        return list(zip(picked["asset_1"], picked["asset_2"]))

    out: List[Tuple[str, str]] = []
    counts: Dict[str, int] = {}
    for r in pool.itertuples(index=False):
        a1 = str(r.asset_1).upper()
        a2 = str(r.asset_2).upper()
        if counts.get(a1, 0) >= max_pairs_per_asset or counts.get(a2, 0) >= max_pairs_per_asset:
            continue
        out.append((a1, a2))
        counts[a1] = counts.get(a1, 0) + 1
        counts[a2] = counts.get(a2, 0) + 1
        if len(out) >= top_n:
            break
    return out


def _context_cache_key(
    cfg: BatchConfig,
    params: StrategyParams,
    universes: List[str],
    scans: Optional[pd.DataFrame],
) -> Tuple:
    start = pd.to_datetime(cfg.start_date).normalize()
    end = pd.to_datetime(cfg.end_date).normalize()

    signal_space = str(params.signal_space).strip().lower()
    panel_buffer_bdays = 400
    if signal_space == "idio_pca":
        panel_buffer_bdays = max(
            panel_buffer_bdays,
            int(params.pca_signal_window) + int(params.z_window) + 25,
        )

    scan_sig = _scan_frame_signature(scans) if scans is not None else _scanner_files_signature(cfg, universes)

    return (
        "global_daily",
        str(Path(cfg.data_path).resolve()),
        str(start.date()),
        str(end.date()),
        tuple(sorted(set(universes))),
        int(params.exec_lag_days),
        _normalize_scan_frequency(params.scan_frequency),
        _normalize_scan_weekday(params.scan_weekday),
        int(params.top_n_candidates),
        str(params.selection_mode).strip().lower(),
        _selection_variant_for_mode(params),
        round(float(params.selection_winsor_quantile), 6),
        round(float(params.selection_stability_penalty), 6),
        round(float(params.selection_half_life_penalty), 6),
        round(float(params.selection_speed_penalty), 6),
        round(float(params.selection_distance_weight), 6),
        round(float(params.selection_corr_penalty), 6),
        _normalize_eligibility_labels(tuple(params.eligibility_labels)),
        None if params.min_corr_12m is None else float(params.min_corr_12m),
        None if params.max_half_life_6m is None else float(params.max_half_life_6m),
        None if params.max_beta_std is None else float(params.max_beta_std),
        None if params.min_spread_std_6m is None else float(params.min_spread_std_6m),
        None if params.min_n_valid_windows is None else int(params.min_n_valid_windows),
        int(params.max_pairs_per_asset),
        signal_space,
        int(params.pca_signal_window),
        int(params.pca_signal_components),
        int(params.pca_signal_min_assets),
        int(panel_buffer_bdays),
        scan_sig,
    )


def _build_global_context(
    key: Tuple,
    cfg: BatchConfig,
    params: StrategyParams,
    universes: List[str],
    scans: Optional[pd.DataFrame],
) -> Optional[_GlobalRunContext]:
    scans_df = _load_scans(cfg, universes, scans)
    if scans_df.empty:
        return None

    start = pd.to_datetime(cfg.start_date).normalize()
    end = pd.to_datetime(cfg.end_date).normalize()

    signal_space = str(params.signal_space).strip().lower()

    extra = int(params.exec_lag_days) + 5
    if MAX_SCAN_AGE_BDAYS is not None:
        extra += int(MAX_SCAN_AGE_BDAYS)

    scans_df = scans_df[
        (scans_df["scan_date"] >= start - BDay(max(5, extra)))
        & (scans_df["scan_date"] <= end)
    ].copy()
    if scans_df.empty:
        return None

    scans_df["scan_date"] = pd.to_datetime(scans_df["scan_date"]).dt.normalize()
    scans_df["asset_1"] = scans_df["asset_1"].astype(str).str.upper()
    scans_df["asset_2"] = scans_df["asset_2"].astype(str).str.upper()
    scans_df = _apply_scan_schedule(scans_df, params)
    if scans_df.empty:
        return None

    scans_by_date = {d: g for d, g in scans_df.groupby("scan_date", sort=True)}
    scan_dates_idx = pd.DatetimeIndex(sorted(scans_by_date.keys())).sort_values()

    all_assets = (
        pd.concat([scans_df["asset_1"], scans_df["asset_2"]], ignore_index=True)
        .astype(str).str.upper().unique().tolist()
    )

    panel_buffer_bdays = 400
    if signal_space == "idio_pca":
        panel_buffer_bdays = max(
            panel_buffer_bdays,
            int(params.pca_signal_window) + int(params.z_window) + 25,
        )

    price_df = build_price_panel(
        cfg=cfg,
        assets=all_assets,
        start=start,
        end=end,
        buffer_bdays=panel_buffer_bdays,
    )
    if price_df.empty:
        return None

    trade_dates = pd.bdate_range(start=start, end=end).normalize()
    eval_dates = scan_dates_idx[(scan_dates_idx >= start) & (scan_dates_idx <= end)]
    if len(eval_dates) == 0:
        return None

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

    pca_thresh_series = (
        pca_mode.expanding(min_periods=252).quantile(PCA_Q)
        if not pca_mode.dropna().empty else None
    )
    vol_thresh_series = (
        mkt_vol.expanding(min_periods=252).quantile(MKT_VOL_Q)
        if not mkt_vol.dropna().empty else None
    )

    mode_vals = pca_mode.to_numpy(dtype=float)
    vol_vals = mkt_vol.to_numpy(dtype=float)
    if pca_thresh_series is None:
        mode_th_vals = np.full_like(mode_vals, np.nan, dtype=float)
    else:
        mode_th_vals = pca_thresh_series.to_numpy(dtype=float)
    if vol_thresh_series is None:
        vol_th_vals = np.full_like(vol_vals, np.nan, dtype=float)
    else:
        vol_th_vals = vol_thresh_series.to_numpy(dtype=float)

    active = (
        np.isfinite(mode_vals)
        & np.isfinite(vol_vals)
        & np.isfinite(mode_th_vals)
        & np.isfinite(vol_th_vals)
    )
    blocked = active & (mode_vals >= mode_th_vals) & (vol_vals >= vol_th_vals)
    pct_blocked = float(blocked.sum() / active.sum()) if int(active.sum()) > 0 else np.nan

    ranked_pairs_by_scan_date: Dict[pd.Timestamp, List[Tuple[str, str]]] = {}
    top_n = int(params.top_n_candidates)
    for scan_dt, df_day in scans_by_date.items():
        selected = _select_ranked_pairs_for_scan_day(df_day=df_day, params=params, top_n=top_n)
        if selected:
            ranked_pairs_by_scan_date[scan_dt] = selected

    ranked_pairs_by_date: Dict[pd.Timestamp, List[Tuple[str, str]]] = {}
    scan_date_by_trade_date: Dict[pd.Timestamp, pd.Timestamp] = {}
    scan_targets = pd.DatetimeIndex(trade_dates - BDay(int(params.exec_lag_days))).normalize()
    scan_dates_ns = scan_dates_idx.view("i8")
    target_ns = scan_targets.view("i8")
    pos = np.searchsorted(scan_dates_ns, target_ns, side="right") - 1

    for i, dt in enumerate(trade_dates):
        if blocked[i]:
            ranked_pairs_by_date[dt] = []
            scan_date_by_trade_date[dt] = pd.NaT
            continue

        p = int(pos[i])
        if p < 0:
            ranked_pairs_by_date[dt] = []
            scan_date_by_trade_date[dt] = pd.NaT
            continue

        scan_dt = pd.Timestamp(scan_dates_idx[p]).normalize()
        if MAX_SCAN_AGE_BDAYS is not None:
            if scan_dt < (scan_targets[i] - BDay(int(MAX_SCAN_AGE_BDAYS))).normalize():
                ranked_pairs_by_date[dt] = []
                scan_date_by_trade_date[dt] = pd.NaT
                continue

        ranked_pairs_by_date[dt] = ranked_pairs_by_scan_date.get(scan_dt, [])
        scan_date_by_trade_date[dt] = scan_dt

    all_ranked_pairs = sorted(
        {
            (a1, a2)
            for pairs in ranked_pairs_by_date.values()
            for (a1, a2) in pairs
        }
    )

    signal_price_panel: Optional[pd.DataFrame] = None
    if signal_space == "idio_pca":
        signal_price_panel = _build_idio_price_panel_from_raw_prices(
            price_df=price_df,
            pca_window=int(params.pca_signal_window),
            n_components=int(params.pca_signal_components),
            min_assets=int(params.pca_signal_min_assets),
        )
        if signal_price_panel.empty or float(signal_price_panel.notna().sum().sum()) <= 0:
            return None
    else:
        signal_price_panel = price_df

    if all_ranked_pairs:
        needed_assets = sorted({a for pair in all_ranked_pairs for a in pair})
        signal_price_panel = signal_price_panel.reindex(columns=needed_assets)

    asset_log_offsets: Dict[str, float] = {}
    if signal_space == "raw":
        data_path_str = str(cfg.data_path)
        for asset in signal_price_panel.columns:
            try:
                s = _load_asset_prices(asset, data_path_str)
            except Exception:
                continue
            if s.empty:
                continue
            first_px = float(s.iloc[0])
            if np.isfinite(first_px) and first_px > 0.0:
                asset_log_offsets[asset] = float(np.log(first_px))

    return _GlobalRunContext(
        key=key,
        start=start,
        end=end,
        trade_dates=trade_dates,
        ranked_pairs_by_date=ranked_pairs_by_date,
        scan_date_by_trade_date=scan_date_by_trade_date,
        all_ranked_pairs=all_ranked_pairs,
        signal_price_panel=signal_price_panel,
        asset_log_offsets=asset_log_offsets,
        pct_blocked=pct_blocked,
        signal_space=signal_space,
    )


def _get_or_build_global_context(
    cfg: BatchConfig,
    params: StrategyParams,
    universes: List[str],
    scans: Optional[pd.DataFrame],
) -> Optional[_GlobalRunContext]:
    key = _context_cache_key(cfg, params, universes, scans)
    with _CACHE_LOCK:
        cached = _lru_get(_GLOBAL_CONTEXT_CACHE, key)
    if cached is not None:
        return cached

    ctx = _build_global_context(key, cfg, params, universes, scans)
    if ctx is None:
        return None

    with _CACHE_LOCK:
        _lru_put(_GLOBAL_CONTEXT_CACHE, key, ctx, _CONTEXT_CACHE_MAXSIZE)
    return ctx


def _build_pair_states_for_window(ctx: _GlobalRunContext, z_window: int) -> Dict[str, pd.DataFrame]:
    panel = ctx.signal_price_panel
    if panel is None or panel.empty or not ctx.all_ranked_pairs:
        return {}

    win = int(z_window)
    warmup_days = int(win + 10)
    warmup_ns = np.int64(warmup_days) * _NS_PER_DAY

    needed_start = (ctx.start - pd.Timedelta(days=warmup_days)).normalize()
    panel = panel.loc[needed_start:ctx.end]
    if panel.empty:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    min_pair_rows = int(win + 5)

    for a1, a2 in ctx.all_ranked_pairs:
        if a1 not in panel.columns or a2 not in panel.columns:
            continue

        px = panel[[a1, a2]]
        p1 = px[a1].to_numpy(dtype=float)
        p2 = px[a2].to_numpy(dtype=float)
        valid = np.isfinite(p1) & np.isfinite(p2) & (p1 > 0.0) & (p2 > 0.0)
        if not bool(valid.any()):
            continue

        idx = px.index[valid]
        if len(idx) < min_pair_rows:
            continue

        if ctx.signal_space == "raw":
            off1 = float(ctx.asset_log_offsets.get(a1, 0.0))
            off2 = float(ctx.asset_log_offsets.get(a2, 0.0))
            y = pd.Series(np.log(p1[valid]) - off1, index=idx, dtype=float)
            x = pd.Series(np.log(p2[valid]) - off2, index=idx, dtype=float)
        else:
            y = pd.Series(np.log(p1[valid]), index=idx, dtype=float)
            x = pd.Series(np.log(p2[valid]), index=idx, dtype=float)

        x_prev = x.shift(1)
        y_prev = y.shift(1)

        sum_x_prev = x_prev.rolling(win, min_periods=win).sum()
        sum_y_prev = y_prev.rolling(win, min_periods=win).sum()
        sum_x2_prev = (x_prev * x_prev).rolling(win, min_periods=win).sum()
        sum_xy_prev = (x_prev * y_prev).rolling(win, min_periods=win).sum()

        den_prev = sum_x2_prev - (sum_x_prev * sum_x_prev) / float(win)
        num_prev = sum_xy_prev - (sum_x_prev * sum_y_prev) / float(win)
        beta = num_prev / den_prev
        beta = beta.mask((den_prev <= 0.0) & den_prev.notna(), 1.0)

        sum_x = x.rolling(win, min_periods=win).sum()
        sum_y = y.rolling(win, min_periods=win).sum()
        sum_x2 = (x * x).rolling(win, min_periods=win).sum()
        sum_y2 = (y * y).rolling(win, min_periods=win).sum()
        sum_xy = (x * y).rolling(win, min_periods=win).sum()

        mean_x = sum_x / float(win)
        mean_y = sum_y / float(win)
        var_x = (sum_x2 - (sum_x * sum_x) / float(win)) / float(win - 1)
        var_y = (sum_y2 - (sum_y * sum_y) / float(win)) / float(win - 1)
        cov_xy = (sum_xy - (sum_x * sum_y) / float(win)) / float(win - 1)

        spread = y - beta * x
        spread_mean = mean_y - beta * mean_x
        spread_var = var_y + (beta * beta) * var_x - 2.0 * beta * cov_xy
        spread_var = spread_var.mask(spread_var < 0.0, np.nan)
        spread_std = np.sqrt(spread_var)

        z = (spread - spread_mean) / spread_std
        z = z.mask((spread_std <= 0.0) & spread_std.notna(), np.nan)

        idx_ns = idx.view("i8")
        left = np.searchsorted(idx_ns, idx_ns - warmup_ns, side="left")
        pos = np.arange(len(idx_ns), dtype=np.int64)
        count_total = pos - left + 1
        count_prev = count_total - 1
        state_available = (count_total >= min_pair_rows) & (count_prev >= win)

        pid = f"{a1}_{a2}"
        out[pid] = pd.DataFrame(
            {
                "pair_id": pid,
                "asset_1": a1,
                "asset_2": a2,
                "y": y.astype(float),
                "x": x.astype(float),
                "beta": beta.astype(float),
                "spread": spread.astype(float),
                "spread_std": spread_std.astype(float),
                "z": z.astype(float),
                "state_available": state_available,
            },
            index=idx,
        )

    return out


def _get_or_build_pair_states_for_window(ctx: _GlobalRunContext, z_window: int) -> Dict[str, pd.DataFrame]:
    key = (ctx.key, int(z_window))
    with _CACHE_LOCK:
        cached = _lru_get(_PAIR_STATE_CACHE, key)
    if cached is not None:
        return cached

    built = _build_pair_states_for_window(ctx, z_window)
    with _CACHE_LOCK:
        _lru_put(_PAIR_STATE_CACHE, key, built, _PAIR_STATE_CACHE_MAXSIZE)
    return built


def run_global_ranking_daily_portfolio(
    cfg: BatchConfig,
    params: StrategyParams,
    universes: List[str],
    scans: Optional[pd.DataFrame] = None,
) -> Dict:
    signal_space = str(params.signal_space).strip().lower()
    selection_mode = str(params.selection_mode).strip().lower()
    selection_variant = _selection_variant_for_mode(params)
    scan_frequency = _normalize_scan_frequency(params.scan_frequency)
    scan_weekday = _normalize_scan_weekday(params.scan_weekday)
    if signal_space not in {"raw", "idio_pca"}:
        raise ValueError(f"Unsupported signal_space='{params.signal_space}'. Use 'raw' or 'idio_pca'.")
    if selection_mode not in {"legacy", "composite_quality"}:
        raise ValueError(
            f"Unsupported selection_mode='{params.selection_mode}'. Use 'legacy' or 'composite_quality'."
        )
    if signal_space == "idio_pca":
        if int(params.pca_signal_window) < 20:
            raise ValueError("pca_signal_window must be >= 20 for signal_space='idio_pca'.")
        if int(params.pca_signal_components) < 1:
            raise ValueError("pca_signal_components must be >= 1 for signal_space='idio_pca'.")
        if int(params.pca_signal_min_assets) < 3:
            raise ValueError("pca_signal_min_assets must be >= 3 for signal_space='idio_pca'.")
    if params.pair_return_cap is not None and float(params.pair_return_cap) <= 0.0:
        raise ValueError("pair_return_cap must be > 0 when provided.")
    if params.trade_return_isolated_cap is not None and float(params.trade_return_isolated_cap) <= 0.0:
        raise ValueError("trade_return_isolated_cap must be > 0 when provided.")
    if params.portfolio_vol_target is not None and float(params.portfolio_vol_target) <= 0.0:
        raise ValueError("portfolio_vol_target must be > 0 when provided.")
    if int(params.portfolio_vol_lookback) < 5:
        raise ValueError("portfolio_vol_lookback must be >= 5.")
    if float(params.portfolio_vol_max_scale) <= 0.0:
        raise ValueError("portfolio_vol_max_scale must be > 0.")

    ctx = _get_or_build_global_context(cfg=cfg, params=params, universes=universes, scans=scans)
    if ctx is None:
        return {}

    pair_state_cache = _get_or_build_pair_states_for_window(ctx, int(params.z_window))

    def get_ranked_pairs(dt: pd.Timestamp) -> List[Tuple[str, str]]:
        dt = pd.to_datetime(dt).normalize()
        return ctx.ranked_pairs_by_date.get(dt, [])

    def get_pair_state(dt: pd.Timestamp, pairs: List[Tuple[str, str]]) -> Dict[str, pd.DataFrame]:
        dt = pd.to_datetime(dt).normalize()
        out: Dict[str, pd.DataFrame] = {}
        for a1, a2 in pairs:
            pid = f"{a1.upper()}_{a2.upper()}"
            dfp = pair_state_cache.get(pid)
            if dfp is None:
                continue
            if dt not in dfp.index:
                continue
            if not bool(dfp.at[dt, "state_available"]):
                continue
            out[pid] = dfp
        return out

    res = run_daily_portfolio_engine(
        params=params,
        start=ctx.start,
        end=ctx.end,
        get_ranked_pairs=get_ranked_pairs,
        get_pair_state=get_pair_state,
    )
    if not res:
        return {}

    equity = res["equity"].copy()
    trades = res["trades"].copy()
    diagnostics = res.get("diagnostics", pd.DataFrame()).copy()
    entry_filter_summary = res.get("entry_filter_summary", pd.DataFrame()).copy()
    anomaly_flags = list(res.get("anomaly_flags", []))

    trade_dates = pd.DatetimeIndex(ctx.trade_dates).normalize()
    trade_pos = {pd.Timestamp(dt).normalize(): i for i, dt in enumerate(trade_dates)}
    scan_usage = pd.DataFrame({"trade_date": trade_dates})
    scan_usage["scan_target_date"] = pd.DatetimeIndex(
        trade_dates - BDay(int(params.exec_lag_days))
    ).normalize()
    scan_usage["applied_scan_date"] = [
        pd.to_datetime(ctx.scan_date_by_trade_date.get(pd.Timestamp(dt).normalize(), pd.NaT))
        for dt in trade_dates
    ]
    scan_usage["scan_age_bdays"] = [
        (
            float(trade_pos[pd.Timestamp(dt).normalize()] - trade_pos[pd.Timestamp(sd).normalize()])
            if pd.notna(sd) and pd.Timestamp(sd).normalize() in trade_pos
            else np.nan
        )
        for dt, sd in zip(scan_usage["trade_date"], scan_usage["applied_scan_date"])
    ]
    scan_usage["lookahead_ok"] = [
        (pd.isna(sd) or pd.Timestamp(sd).normalize() < pd.Timestamp(dt).normalize())
        for dt, sd in zip(scan_usage["trade_date"], scan_usage["applied_scan_date"])
    ]
    n_lookahead_violations = int((~scan_usage["lookahead_ok"]).sum())
    avg_scan_age_bdays = float(scan_usage["scan_age_bdays"].mean()) if int(scan_usage["scan_age_bdays"].notna().sum()) > 0 else np.nan

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
    cagr = (final_eq ** (252 / n) - 1.0) if (n > 0 and final_eq > 0.0) else np.nan
    vol = float(returns.std(ddof=1)) if len(returns) > 1 else np.nan
    sharpe = float(np.sqrt(252) * returns.mean() / vol) if (vol is not None and vol > 0) else np.nan
    mdd = float((equity["equity"] / equity["equity"].cummax() - 1).min())
    max_abs_daily_ret = float(np.abs(returns).max()) if len(returns) > 0 else np.nan
    p99_abs_daily_ret = float(np.abs(returns).quantile(0.99)) if len(returns) > 0 else np.nan

    max_abs_pair_ret_raw = np.nan
    max_vol_scale = np.nan
    if isinstance(diagnostics, pd.DataFrame) and not diagnostics.empty:
        if "max_abs_pair_ret_raw" in diagnostics.columns:
            s = pd.to_numeric(diagnostics["max_abs_pair_ret_raw"], errors="coerce")
            max_abs_pair_ret_raw = float(s.max()) if int(s.notna().sum()) > 0 else np.nan
        if "vol_scale" in diagnostics.columns:
            s = pd.to_numeric(diagnostics["vol_scale"], errors="coerce")
            max_vol_scale = float(s.max()) if int(s.notna().sum()) > 0 else np.nan

    anomaly_reasons = []
    if any(str(x).strip() for x in anomaly_flags):
        anomaly_reasons.extend([str(x) for x in anomaly_flags if str(x).strip()])
    if np.isfinite(mdd) and mdd <= -1.0:
        anomaly_reasons.append("drawdown_below_-100pct")
    if np.isfinite(max_abs_daily_ret) and max_abs_daily_ret > 0.35:
        anomaly_reasons.append("extreme_daily_return_gt_35pct")
    if np.isfinite(max_abs_pair_ret_raw) and max_abs_pair_ret_raw > 0.20:
        anomaly_reasons.append("extreme_pair_mtm_raw_gt_20pct")
    anomaly_reasons = sorted(set(anomaly_reasons))

    stats = {
        "Final Equity": round(final_eq, 2),
        "CAGR": round(float(cagr), 3) if not np.isnan(cagr) else np.nan,
        "Sharpe": round(float(sharpe), 2) if not np.isnan(sharpe) else np.nan,
        "Max Drawdown": round(mdd, 3),
        "Nb Trades": int(len(trades)) if isinstance(trades, pd.DataFrame) else 0,
        "Signal space": ctx.signal_space,
        "Entry mode": str(params.entry_mode).strip().lower(),
        "Scan frequency": scan_frequency,
        "Scan weekday": scan_weekday,
        "Selection mode": selection_mode,
        "Selection variant": selection_variant,
        "Selection labels": ",".join(_normalize_eligibility_labels(tuple(params.eligibility_labels))),
        "Avg scan age (bdays)": round(avg_scan_age_bdays, 2) if np.isfinite(avg_scan_age_bdays) else np.nan,
        "Lookahead violations": n_lookahead_violations,
        "Max pairs per asset": int(params.max_pairs_per_asset),
        "Selection winsor q": float(params.selection_winsor_quantile),
        "Selection stability penalty": float(params.selection_stability_penalty),
        "Selection half-life penalty": float(params.selection_half_life_penalty),
        "Selection speed penalty": float(params.selection_speed_penalty),
        "Selection distance weight": float(params.selection_distance_weight),
        "Selection corr penalty": float(params.selection_corr_penalty),
        "Pair return cap": float(params.pair_return_cap) if params.pair_return_cap is not None else np.nan,
        "Portfolio vol target": (
            float(params.portfolio_vol_target) if params.portfolio_vol_target is not None else np.nan
        ),
        "Portfolio vol lookback": int(params.portfolio_vol_lookback),
        "Portfolio vol max scale": float(params.portfolio_vol_max_scale),
        "PCA window": PCA_WINDOW,
        "PCA q": PCA_Q,
        "PCA min assets": PCA_MIN_ASSETS,
        "Mkt vol window": MKT_VOL_WINDOW,
        "Mkt vol q": MKT_VOL_Q,
        "% days blocked (when active)": round(float(ctx.pct_blocked), 3) if np.isfinite(ctx.pct_blocked) else np.nan,
        "Max scan age (bdays)": MAX_SCAN_AGE_BDAYS,
        "Max abs daily return": round(max_abs_daily_ret, 4) if np.isfinite(max_abs_daily_ret) else np.nan,
        "P99 abs daily return": round(p99_abs_daily_ret, 4) if np.isfinite(p99_abs_daily_ret) else np.nan,
        "Max abs pair mtm raw": round(max_abs_pair_ret_raw, 4) if np.isfinite(max_abs_pair_ret_raw) else np.nan,
        "Max vol scale": round(max_vol_scale, 3) if np.isfinite(max_vol_scale) else np.nan,
        "Anomaly flag": bool(anomaly_reasons),
        "Anomaly reasons": ";".join(anomaly_reasons),
    }
    if signal_space == "idio_pca":
        stats["Signal PCA window"] = int(params.pca_signal_window)
        stats["Signal PCA components"] = int(params.pca_signal_components)
        stats["Signal PCA min assets"] = int(params.pca_signal_min_assets)

    return {
        "equity": equity,
        "monthly": monthly,
        "trades": trades,
        "stats": stats,
        "diagnostics": diagnostics,
        "entry_filter_summary": entry_filter_summary,
        "scan_usage": scan_usage,
        "anomaly_flags": anomaly_reasons,
    }
