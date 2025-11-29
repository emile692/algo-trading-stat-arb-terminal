import numpy as np
import pandas as pd
from utils.synthetic import simulate_cointegrated_assets, calibrate_params_from_pair
from utils.metrics import compute_zscore
from utils.backtest import backtest_pair, walk_forward_beta_spread_zscore


# ============================================================
# Sharp ratio util
# ============================================================
def compute_sharpe(equity):
    rets = np.diff(equity)
    if len(rets) < 2:
        return 0.0
    std = np.std(rets)
    if std == 0:
        return 0.0
    return float(np.mean(rets) / std * np.sqrt(252))


# ============================================================
# Param grid sampling
# ============================================================
def sample_parameters(n, z_entry_range, z_exit_range, z_window_range, wf_train_range):
    params = []
    for _ in range(n):
        wf_train = int(np.random.randint(*wf_train_range))
        params.append({
            "z_entry": float(np.random.uniform(*z_entry_range)),
            "z_exit": float(np.random.uniform(*z_exit_range)),
            "z_window": int(np.random.randint(*z_window_range)),
            "wf_train": wf_train,
            "wf_test": int(max(5, wf_train * 0.3))
        })
    return params


# ============================================================
# Real backtest
# ============================================================
def run_single_backtest_real(merged, asset1, asset2, param):
    spread_bt, zscore_bt, beta_wf = walk_forward_beta_spread_zscore(
        merged[f"norm_{asset1}"],
        merged[f"norm_{asset2}"],
        train=param["wf_train"],
        test=param["wf_test"],
        z_window=param["z_window"],
    )

    equity, trades, pnl_list = backtest_pair(
        spread_bt,
        zscore_bt,
        merged[f"norm_{asset1}"],
        merged[f"norm_{asset2}"],
        beta=1,
        z_entry=param["z_entry"],
        z_exit=param["z_exit"],
        z_stop=param["z_entry"] * 2,
        fees=0.0002
    )

    return compute_sharpe(equity)


# ============================================================
# Synthetic backtest using A/B simulated
# ============================================================
def run_single_backtest_synth(A, B, param, beta):
    A = pd.Series(A)
    B = pd.Series(B)
    spread_sim = A - beta * B
    zscore_sim = compute_zscore(spread_sim, window=param["z_window"]).fillna(0)

    equity, trades, pnl_list = backtest_pair(
        spread_sim,
        zscore_sim,
        A,
        B,
        beta=beta,
        z_entry=param["z_entry"],
        z_exit=param["z_exit"],
        z_stop=param["z_entry"] * 2,
        fees=0.0002
    )

    return compute_sharpe(equity)


# ============================================================
# Full robust optimization
# ============================================================
def run_full_optimization(
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
    seed=42,
):
    np.random.seed(seed)

    # 1) param grid
    params_list = sample_parameters(
        n_param,
        z_entry_range,
        z_exit_range,
        z_window_range,
        wf_train_range,
    )

    # 2) calibrate Heston + OU
    calib = calibrate_params_from_pair(df1, df2, spread, beta)

    # 3) generate synthetic pairs
    n_steps = len(merged)
    synth_pairs = []
    for _ in range(n_synth):
        A, B, X, S, v = simulate_cointegrated_assets(
            n_steps,
            beta=beta,
            X0=calib["X0"], v0=calib["v0"], mu=calib["mu"],
            theta=calib["theta"], kappa=calib["kappa"],
            xi=calib["xi"], rho=calib["rho"],
            mu_s=calib["mu_s"], theta_s=calib["theta_s"],
            sigma_s=calib["sigma_s"], S0=calib["S0"]
        )
        synth_pairs.append((A, B))

    # 4) optimization loop
    results = []

    for param in params_list:

        # real sharpe
        sharpe_real = run_single_backtest_real(merged, asset1, asset2, param)

        # synthetic sharpes
        sharpes_synth = []
        for (A_s, B_s) in synth_pairs:
            sharpe_sim = run_single_backtest_synth(A_s, B_s, param, beta)
            sharpes_synth.append(sharpe_sim)

        sharpes_synth = np.array(sharpes_synth)

        results.append({
            **param,
            "Sharpe_real": float(sharpe_real),
            "Sharpe_median": float(np.median(sharpes_synth)),
            "Sharpe_min": float(np.min(sharpes_synth)),
            "Sharpe_std": float(np.std(sharpes_synth)),
            "Robustness": float(np.median(sharpes_synth) - np.std(sharpes_synth)),
        })

    df_res = pd.DataFrame(results).sort_values(
        ["Robustness", "Sharpe_real", "Sharpe_median"],
        ascending=False
    ).reset_index(drop=True)

    return df_res
