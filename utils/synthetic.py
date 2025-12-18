import numpy as np
import pandas as pd

from utils.metrics import compute_half_life

def stationary_bootstrap(series, block_prob=0.1, size=None):
    """
    Generate a synthetic series using stationary bootstrap
    (Politis & Romano, 1994).

    series: 1D numpy array or pandas series
    block_prob: probability of starting a new block (p)
    size: length of synthetic output
    """
    series = np.asarray(series)
    T = len(series)
    if size is None:
        size = T

    synthetic = np.zeros(size)
    idx = np.random.randint(0, T)  # initial point

    for t in range(size):
        synthetic[t] = series[idx]

        # continue block or start new block?
        if np.random.rand() < block_prob:
            idx = np.random.randint(0, T)  # new block
        else:
            idx = (idx + 1) % T  # continue block

    return synthetic


def generate_synthetic_paths(n_paths: int, n_steps: int, mu: float, sigma: float, s0: float):
    """
    Génère des trajectoires de prix avec un mouvement brownien géométrique.
    """

    dt = 1.0
    paths = np.zeros((n_paths, n_steps))

    for i in range(n_paths):
        path = np.zeros(n_steps)
        path[0] = s0

        for t in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            path[t] = path[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

        paths[i] = path

    return paths


def generate_ou_paths(n_paths, n_steps, mu, theta, sigma, s0, dt=1.0):
    sigma = sigma * 3
    theta = theta * 2
    paths = np.zeros((n_paths, n_steps))
    for i in range(n_paths):
        S = s0
        for t in range(n_steps):
            dW = np.random.normal(0, np.sqrt(dt))
            S = S + theta * (mu - S) * dt + sigma * dW
            paths[i, t] = S
    return paths


def simulate_heston(
    n_steps,
    X0=100.0,
    v0=0.04,
    mu=0.05,
    kappa=1.5,
    theta=0.04,
    xi=0.3,
    rho=-0.7,
    dt=1/252
):
    """
    Retourne X_t et v_t simulés sous Heston
    (schéma full-truncation pour éviter les NaN)
    """
    X = np.zeros(n_steps)
    v = np.zeros(n_steps)

    X[0] = X0
    v[0] = max(v0, 1e-8)  # évite v = 0 ou négatif

    for t in range(1, n_steps):
        z1 = np.random.normal()
        z2 = np.random.normal()

        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

        # on tronque v[t-1] avant de l'utiliser
        v_prev = max(v[t-1], 1e-8)

        # mise à jour de la variance (full truncation)
        dv = kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev * dt) * w2
        v_new = v_prev + dv
        v[t] = max(v_new, 1e-8)

        # mise à jour du prix avec v_prev (non négatif)
        X[t] = X[t-1] * np.exp(
            (mu - 0.5 * v_prev) * dt + np.sqrt(v_prev * dt) * w1
        )

    return X, v



def simulate_ou(n_steps, mu_s, theta_s, sigma_s, S0=0, dt=1.0):
    """
    OU en unité 'bar' (dt=1.0 par défaut), cohérent avec compute_half_life()
    et avec theta_s = log(2)/half_life (half_life en bars).
    """
    S = np.zeros(n_steps)
    S[0] = S0

    for t in range(1, n_steps):
        dW = np.random.normal() * np.sqrt(dt)
        S[t] = S[t-1] + theta_s * (mu_s - S[t-1]) * dt + sigma_s * dW

    return S



def calibrate_params_from_pair(df1, df2, spread, beta: float) -> dict:
    """
    Calibre des paramètres simples pour la génération synthétique.
    Robuste si spread est ndarray, et robuste aux NaN.
    """
    spread_s = pd.Series(spread).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(spread_s) < 50:
        # fallback très conservateur
        return {
            "X0": float(spread_s.iloc[-1]) if len(spread_s) else 0.0,
            "v0": 0.0,
            "mu": float(spread_s.mean()) if len(spread_s) else 0.0,
            "theta": 0.05,
            "kappa": 1.0,
            "xi": 0.10,
            "rho": 0.0,
            "mu_s": 1.0,
            "theta_s": 0.05,
            "sigma_s": 0.01,
            "S0": float(spread_s.iloc[-1]) if len(spread_s) else 0.0,
        }

    # Half-life => theta_s
    hl = compute_half_life(spread_s)
    theta_s = (np.log(2) / hl) if (hl is not None and np.isfinite(hl) and hl > 0) else 0.05
    theta_s = float(np.clip(theta_s, 1e-4, 2.0))

    # Vol résiduelle (sigma_s) : std des incréments
    sigma_s = float(spread_s.diff().dropna().std())
    if not np.isfinite(sigma_s) or sigma_s <= 0:
        sigma_s = float(spread_s.std()) if np.isfinite(spread_s.std()) and spread_s.std() > 0 else 0.01
    sigma_s = float(np.clip(sigma_s, 1e-6, 10.0))

    # Niveau moyen
    mu_s = float(spread_s.mean())

    # S0
    S0 = float(spread_s.iloc[-1])

    # Partie cointegration “X/v” : on reste simple et stable
    X0 = S0
    v0 = 0.0
    mu = mu_s
    theta = 0.05
    kappa = 1.0
    xi = 0.10
    rho = 0.0

    return {
        "X0": X0,
        "v0": v0,
        "mu": mu,
        "theta": theta,
        "kappa": kappa,
        "xi": xi,
        "rho": rho,
        "mu_s": mu_s,
        "theta_s": theta_s,
        "sigma_s": sigma_s,
        "S0": S0,
    }



def simulate_cointegrated_assets(
    n_steps,
    beta,
    # market factor (Heston)
    X0, v0, mu, theta, kappa, xi, rho,
    # OU spread (déjà en unités de spread réel, donc en log-space)
    mu_s, theta_s, sigma_s, S0,
    dt_mkt=1/252,
    dt_ou=1.0,
):
    """
    Produit A, B dans le même espace que le backtest réel :
      - X est simulé en PRIX via Heston, puis converti en log et normalisé
      - spread tradé = A - beta*B = S (OU)
    """

    # 1) Heston -> prix
    X_price, v = simulate_heston(
        n_steps, X0=X0, v0=v0, mu=mu,
        kappa=kappa, theta=theta,
        xi=xi, rho=rho, dt=dt_mkt
    )

    # 2) Conversion en log + normalisation (comme df["norm"])
    X_log = np.log(np.maximum(X_price, 1e-12))
    X_norm = X_log - X_log[0]   # même convention que ton "norm" réel

    # 3) OU spread en "bars"
    S = simulate_ou(
        n_steps, mu_s, theta_s, sigma_s, S0, dt=dt_ou
    )

    # 4) Reconstruction cointegrée dans le même espace (log-norm)
    B = X_norm
    A = beta * X_norm + S

    return A, B, X_norm, S, v


