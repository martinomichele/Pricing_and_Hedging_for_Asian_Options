import numpy as np
from scipy.stats import norm

def asian_average(S: np.ndarray,
                  kind: str = "arith") -> np.ndarray:
    """
    Compute the per-path average of the simulated prices.
    
    Parameters:
    S : (n_paths, d) array
        Simulated price paths over d monitoring dates (excluding S0 by default).
    kind : {'arith','geom'}
        Type of average for the payoff definition.
    
    Returns:
    A : (n_paths,) array of per-path average.
    """

    if kind == "arith":
        return S.mean(axis=1)
    elif kind == "geom":
        # geometric mean (robust to underflow via logs)
        return np.exp(np.mean(np.log(S), axis=1))


def asian_payoff_paths(S: np.ndarray,
                       K: float,
                       r: float,
                       T: float,
                       option: str = "call",
                       average: str = "arith") -> np.ndarray:
    """
    Discounted Asian option payoff for each simulated path.

    Parameters:
    S : (n_paths, d) array
        Simulated price paths.
    K : float
        Strike.
    r : float
        Risk-free continuously compounded rate.
    T : float
        Time to maturity.
    option : {'call','put'}
        Payoff side.
    average : {'arith','geom'}

    Returns:
    payoffs : (n_paths,) array
        Discounted payoff per path at expiry.
    """
    A = asian_average(S, kind=average)
    if option == "call":
        intrinsic = np.maximum(A - K, 0.0)
    elif option == "put":
        intrinsic = np.maximum(K - A, 0.0)

    discount = np.exp(-r * T)
    return discount * intrinsic


def asian_price_estimate(S: np.ndarray,
                         K: float,
                         r: float,
                         T: float,
                         option: str = "call",
                         average: str = "arith") -> tuple[float, float]:
    """
    Return (mean_price, std_error) from the per-path payoffs in a single batch.
    """
    payoffs = asian_payoff_paths(S, K, r, T, option, average)
    n = payoffs.size
    mean_price = float(np.mean(payoffs))
    se = float(np.std(payoffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return mean_price, se


def geometric_asian_call_price_discrete(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    d: int,
) -> float:
    """
    Closed-form price of a discrete-monitoring geometric-average Asian call under GBM.

    Parameters:
    S0, K, r, sigma, T : floats
        Spot, strike, risk-free (cont. comp.), vol, maturity (years).
    d : int
        Number of monitoring dates (equally spaced). Excludes t=0.

    Returns:
    call_price : float
    """

    a1 = (d + 1.0) / (2.0 * d)
    a2 = (d + 1.0) * (2.0 * d + 1.0) / (6.0 * d * d)

    m  = np.log(S0) + (r - 0.5 * sigma**2) * (a1 * T)
    s2 = (sigma**2) * (a2 * T)
    s  = np.sqrt(s2)

    if s == 0.0:
        G = np.exp(m)
        return float(np.exp(-r * T) * max(G - K, 0.0))

    d1 = (m - np.log(K) + s2) / s
    d2 = d1 - s
    call = np.exp(-r * T) * (np.exp(m + 0.5 * s2) * norm.cdf(d1) - K * norm.cdf(d2))
    return float(call)


