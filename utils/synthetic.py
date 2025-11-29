import numpy as np
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
    """
    X = np.zeros(n_steps)
    v = np.zeros(n_steps)

    X[0] = X0
    v[0] = v0

    # Brownian correlés
    for t in range(1, n_steps):
        z1 = np.random.normal()
        z2 = np.random.normal()
        w1 = z1
        w2 = rho*z1 + np.sqrt(1-rho**2)*z2

        # vol stochastique
        v[t] = np.abs(
            v[t-1]
            + kappa*(theta - v[t-1])*dt
            + xi*np.sqrt(v[t-1]*dt)*w2
        )

        # prix
        X[t] = X[t-1] * np.exp(
            (mu - 0.5*v[t-1])*dt + np.sqrt(v[t-1]*dt)*w1
        )

    return X, v


def simulate_ou(n_steps, mu_s, theta_s, sigma_s, S0=0, dt=1/252):
    S = np.zeros(n_steps)
    S[0] = S0

    for t in range(1, n_steps):
        dW = np.random.normal() * np.sqrt(dt)
        S[t] = S[t-1] + theta_s*(mu_s - S[t-1])*dt + sigma_s*dW

    return S


def calibrate_params_from_pair(df1, df2, spread, beta):
    """
    Calibre automatiquement les paramètres pour la simulation cointegrée.
    df1, df2 : prix réels (dataframes)
    spread : série du spread
    beta : hedge ratio réel
    """
    # === 1) Calibration OU (déjà connu dans ton app)
    mu_s = float(spread.mean())
    sigma_s = float(np.std(spread.diff().dropna()))
    hl = compute_half_life(spread)
    theta_s = np.log(2) / hl if hl and hl > 0 else 0.05
    S0 = float(spread.iloc[-1])

    # === 2) Calibration marché X(t)
    # On approxime X(t) comme moyenne pondérée des deux actifs
    mid = 0.5 * (df1["norm"] + df2["norm"])
    ret = mid.diff().dropna()

    # Drift & Vol
    mu = float(ret.mean())
    vol = float(ret.std())

    # Paramètres Heston simples
    # => v0 = vol^2, theta = v0, kappa et xi constants raisonnables
    X0 = mid.iloc[-1]
    v0 = vol**2
    theta = v0
    kappa = 1.5
    xi = 0.5
    rho = -0.3  # corr prix-vol réaliste

    return {
        "mu_s": mu_s,
        "theta_s": theta_s,
        "sigma_s": sigma_s,
        "S0": S0,
        "beta": beta,
        "X0": X0,
        "v0": v0,
        "mu": mu,
        "theta": theta,
        "kappa": kappa,
        "xi": xi,
        "rho": rho,
    }


def simulate_cointegrated_assets(
    n_steps,
    beta,
    # market factor (Heston)
    X0, v0, mu, theta, kappa, xi, rho,
    # OU spread
    mu_s, theta_s, sigma_s, S0,
    dt=1/252,
):
    # 1) marché (Heston)
    X, v = simulate_heston(
        n_steps, X0=X0, v0=v0, mu=mu,
        kappa=kappa, theta=theta,
        xi=xi, rho=rho, dt=dt
    )

    # 2) spread OU
    S = simulate_ou(
        n_steps, mu_s, theta_s, sigma_s, S0, dt
    )

    # 3) reconstruction cointegrée
    A = X + S
    B = X - beta * S

    return A, B, X, S, v
