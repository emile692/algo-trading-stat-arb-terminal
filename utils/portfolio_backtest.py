# utils/portfolio_backtest.py
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.backtest import walk_forward_beta_spread_zscore, backtest_pair
from utils.metrics import compute_hedge_ratio
from utils.loader import load_price_csv



def _equity_to_returns(equity: pd.Series) -> pd.Series:
    eq = pd.Series(equity).astype(float)
    rets = eq.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rets.name = "ret"
    return rets


def backtest_single_pair_from_config(
    merged: pd.DataFrame,
    asset1: str,
    asset2: str,
    cfg_params: dict,
) -> tuple[pd.Series, pd.Series]:
    """
    Retourne (equity, returns) pour UNE paire, en utilisant les params figés du registre.
    Hypothèse: merged contient norm_{asset} + datetime indexable.
    """
    y = pd.Series(merged[f"norm_{asset1}"]).astype(float)
    x = pd.Series(merged[f"norm_{asset2}"]).astype(float)

    # beta fallback (cohérent, recalculé sur tout l'échantillon si absent)
    beta = float(cfg_params.get("beta", np.nan))
    if not np.isfinite(beta):
        beta = compute_hedge_ratio(y, x)
        beta = float(beta) if np.isfinite(beta) else 1.0

    wf_train = int(cfg_params["wf_train"])
    wf_test = int(cfg_params["wf_test"])
    z_window = int(cfg_params["z_window"])

    spread_bt, zscore_bt, beta_wf = walk_forward_beta_spread_zscore(
        y, x, train=wf_train, test=wf_test, z_window=z_window
    )

    # Backtest (si ton backtest_pair ne supporte pas beta_series, enlève-le ici)
    equity, trades, pnl_list = backtest_pair(
        spread_bt,
        zscore_bt,
        y,
        x,
        beta=beta,
        z_entry=float(cfg_params["z_entry"]),
        z_exit=float(cfg_params["z_exit"]),
        z_stop=float(cfg_params.get("z_stop", float(cfg_params["z_entry"]) * 2)),
        fees=float(cfg_params.get("fees", 0.0002)),
    )

    equity = pd.Series(equity).astype(float)
    rets = _equity_to_returns(equity)
    return equity, rets


def backtest_portfolio_equal_weight(
    registry_rows: list[dict],
    data_path,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Backtest portfolio equal-weight à partir du registry.
    Chaque paire reconstruit son merged localement.
    """

    if not isinstance(registry_rows, list):
        raise TypeError(f"registry_rows must be a list, got {type(registry_rows)}")

    # sécurité: si on reçoit une liste de strings, on crashe proprement
    for i, row in enumerate(registry_rows[:5]):
        if not isinstance(row, dict):
            raise TypeError(
                f"registry_rows[{i}] must be a dict, got {type(row)} (value={row})"
            )

    per_pair_rets = []

    for row in registry_rows:
        asset1 = row["asset1"]
        asset2 = row["asset2"]
        pair_id = row.get("pair_id", f"{asset1}-{asset2}")
        params = row["params"]

        df1 = load_price_csv(asset1, data_path).copy()
        df2 = load_price_csv(asset2, data_path).copy()

        df1["log"] = np.log(df1["close"])
        df2["log"] = np.log(df2["close"])

        df1["norm"] = df1["log"] - df1["log"].iloc[0]
        df2["norm"] = df2["log"] - df2["log"].iloc[0]

        merged = pd.merge(
            df1[["datetime", "norm"]],
            df2[["datetime", "norm"]],
            on="datetime",
            how="inner",
            suffixes=(f"_{asset1}", f"_{asset2}"),
        )

        equity, rets = backtest_single_pair_from_config(
            merged,
            asset1,
            asset2,
            cfg_params=params,
        )

        per_pair_rets.append(pd.Series(rets, name=pair_id))

    if not per_pair_rets:
        return pd.Series(dtype=float), pd.DataFrame()

    ret_mat = pd.concat(per_pair_rets, axis=1).fillna(0.0)
    pf_rets = ret_mat.mean(axis=1)
    pf_equity = (1.0 + pf_rets).cumprod()
    pf_equity.name = "portfolio_equity"

    return pf_equity, ret_mat

