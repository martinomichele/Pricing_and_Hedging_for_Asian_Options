import numpy as np

# FUNCTIONS TO GENERATE GEOMETRIC BROWNIAN MOTION PATHS

def gbm_from_increments(S0: float, r: float, sigma: float, dt: float, eps: np.ndarray) -> np.ndarray:
    """
    Map standard-normal increments eps (n_paths, d) to GBM price paths (n_paths, d) using the exact log-Euler solution for GBM.
    """
    drift = (r - 0.5 * sigma**2) * dt
    incr = drift + sigma * np.sqrt(dt) * eps            
    logS = np.log(S0) + np.cumsum(incr, axis=1)         
    return np.exp(logS)


def bm_levels_to_increments(W: np.ndarray, dt: float) -> np.ndarray:
    """
    Convert BM levels W(t_k) to standard-normal increments eps_k = (W(t_k) - W(t_{k-1})) / sqrt(dt)
    """
    W0 = np.zeros((W.shape[0], 1), dtype=W.dtype)
    dW = np.diff(np.concatenate([W0, W], axis=1), axis=1)
    return dW / np.sqrt(dt)


def path_standard_from_Z(S0: float, r: float, sigma: float, T: float, d: int, Z: np.ndarray) -> np.ndarray:
    """
    STANDARD GENERATION: Considers independent standard increments and build GBM incremaentally.
    """
    assert Z.shape[1] == d
    dt = T / d
    eps = Z
    return gbm_from_increments(S0, r, sigma, dt, eps)


def precompute_pca_levels(T: float, d: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Eigendecomposition (PCA) of Brownian levels covariance C_ij = min(t_i, t_j). 
    Returns eigenvectors V (d,d), sqrt eigenvalues s (d,), and dt.
    Eigenvalues are sorted descending to map largest-variance modes first.
    """
    dt = T / d
    t = (np.arange(1, d + 1) * dt).astype(float)
    C = np.minimum.outer(t, t)
    lam, V = np.linalg.eigh(C)           
    idx = lam.argsort()[::-1]            
    lam = lam[idx]
    V = V[:, idx]
    s = np.sqrt(np.clip(lam, 0.0, None))
    return V, s, dt


def path_pca_from_Z(
    S0: float, r: float, sigma: float, T: float, d: int, Z: np.ndarray,
    V: np.ndarray | None = None, s: np.ndarray | None = None
) -> np.ndarray:
    """
    SPECTRAL/PCA PATH GENERATION: Rearrange increments according to Brownian levels PCA
    """
    assert Z.shape[1] == d
    if V is None or s is None:
        V, s, dt = precompute_pca_levels(T, d)
    else:
        dt = T / d

    A = (V * s)                           
    W = Z @ A.T
    eps = bm_levels_to_increments(W, dt)
    return gbm_from_increments(S0, r, sigma, dt, eps)


def precompute_bridge_order(d: int) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """
    Builds midpoint fill order for a Brownian bridge.
    """
    order: list[tuple[int, int, int]] = []

    def build(a: int, b: int) -> None:
        if b - a <= 1:
            return
        m = (a + b) // 2
        order.append((a, m, b))
        build(a, m)
        build(m, b)

    # start from interval [-1, d-1]: a=-1 denotes t=0 (W(0)=0)
    build(-1, d - 1)

    # conditional variance factors in index units (t_k = k+1; t_{-1}=0)
    vf = []
    for a, m, b in order:
        ta = 0 if a < 0 else (a + 1)
        tm = m + 1
        tb = b + 1
        vf.append((tm - ta) * (tb - tm) / (tb - ta))
    return order, np.asarray(vf, dtype=float)


def path_bridge_from_Z(
    S0: float, r: float, sigma: float, T: float, d: int, Z: np.ndarray,
    bridge_cache: tuple[list[tuple[int, int, int]], np.ndarray] | None = None
) -> np.ndarray:
    """
    BROWNIAN BRIDGE PATH GENERATION.
    """
    assert Z.shape[1] == d
    n_paths = Z.shape[0]
    dt = T / d

    if bridge_cache is None:
        order, var_f = precompute_bridge_order(d)
    else:
        order, var_f = bridge_cache

    # Brownian levels at times t_1..t_d (indices 0..d-1)
    W = np.zeros((n_paths, d), dtype=Z.dtype)

    # 1) Endpoint W(T)
    W[:, d - 1] = np.sqrt(T) * Z[:, 0]

    # 2) Midpoints conditionally
    col = 1
    for (a, m, b), vf in zip(order, var_f):
        Wa = 0.0 if a < 0 else W[:, a]
        Wb = W[:, b]  # guaranteed filled already by construction
        ta = 0 if a < 0 else (a + 1)
        tm = m + 1
        tb = b + 1
        mean = ((tm - ta) * Wb + (tb - tm) * Wa) / (tb - ta)
        std = np.sqrt(vf * dt)
        W[:, m] = mean + std * Z[:, col]
        col += 1

    # 3) Levels → increments → GBM
    eps = bm_levels_to_increments(W, dt)
    return gbm_from_increments(S0, r, sigma, dt, eps)


# GENERAL PATH GENERATION FUNCTIONS WHERE PATH GENERATION METHODS ABOVE (Standard, PCA, Brownian Bridge) CAN BE SELECTED:

def precompute_all(T: float, d: int) -> dict:
    """
    Build a dict with all precomputations for reuse across replicates/methods.
    Keys:
        'pca_V'  : PCA eigenvectors
        'pca_s'  : sqrt eigenvalues
        'bridge' : (order, var_factors) for Brownian bridge
    """
    V, s, _ = precompute_pca_levels(T, d)
    bridge_cache = precompute_bridge_order(d)
    return {'pca_V': V, 'pca_s': s, 'bridge': bridge_cache}


def generate_paths(
    method: str, S0: float, r: float, sigma: float, T: float, d: int,
    Z: np.ndarray, precomp: dict | None = None
) -> np.ndarray:
    """
    method ∈ {'standard','pca','bridge'}
    Z: (n_paths, d) standard normals.
    precomp: optional dict from precompute_all(...) to avoid recomputation.
    """
    method = method.lower()
    if method == 'standard':
        return path_standard_from_Z(S0, r, sigma, T, d, Z)

    elif method == 'pca':
        V = None if precomp is None else precomp.get('pca_V', None)
        s = None if precomp is None else precomp.get('pca_s', None)
        return path_pca_from_Z(S0, r, sigma, T, d, Z, V=V, s=s)

    elif method == 'bridge':
        bridge_cache = None if precomp is None else precomp.get('bridge', None)
        return path_bridge_from_Z(S0, r, sigma, T, d, Z, bridge_cache=bridge_cache)



