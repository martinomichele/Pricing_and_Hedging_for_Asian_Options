import numpy as np
from typing import Iterable, List, Optional, Tuple
from scipy.stats import qmc, norm

# HELPER FUNCTIONS FOR quasi MC sampling and baseline MC sampling

def sobol_points(
    n: int,
    d: int,
    scramble: bool = False,
    seed: Optional[int] = None,
    use_base2: bool = False,
) -> np.ndarray:
    """
    Generate Sobol low-discrepancy points in [0,1)^d.

    Parameters:
    n : int
        Number of points (i.e., number of simulations).
    d : int
        Dimension (i.e. time discretization points).
    scramble : bool, default False
        If True, use Owen scrambling.
    seed : int or None
        Seed for scrambling (ignored if scramble=False).
    use_base2 : bool, default False
        If True, uses engine.random_base2(m) with n = 2**m points (fastest).
        If set True, n must be a power of 2.

    Returns:
    U : (n, d) ndarray
        Points in [0,1)^d.
    """
    eng = qmc.Sobol(d=d, scramble=scramble, seed=seed)

    if use_base2:
        # n must be a power of two
        m = int(np.round(np.log2(n)))
        if 2**m != n:
            raise ValueError("use_base2=True requires n to be a power of 2.")
        U = eng.random_base2(m=m)
    else:
        U = eng.random(n)

    return U


def prn_uniform(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Pseudo-random i.i.d. uniforms in [0,1)^d using PCG64.
    Used as the MC baseline against QMC.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.random((n, d))


def random_digital_shift(U: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """
    Apply a random digital shift mod 1 to any point set U in [0,1)^d.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    d = U.shape[1]
    shift = rng.random(d)
    return (U + shift) % 1.0


def sobol_replicates(
    n: int,
    d: int,
    R: int,
    base_seed: int = 0,
    use_base2: bool = False,
) -> List[np.ndarray]:
    """
    Produce R independent *scrambled* Sobol point sets (RQMC replicates).
    Each replicate uses a different seed.

    Returns:
    reps : list of (n, d) arrays
    """
    reps: List[np.ndarray] = []
    for r in range(R):
        U = sobol_points(
            n=n, d=d, scramble=True,
            seed=base_seed + r, use_base2=use_base2
        )
        reps.append(U)
    return reps


def replicate_with_digital_shift(
    U: np.ndarray, R: int, base_seed: int = 0
) -> List[np.ndarray]:
    """
    Turn one deterministic point set U into R independent RQMC replicates
    using random digital shifts.
    """
    return [random_digital_shift(U, seed=base_seed + r) for r in range(R)]


def u_to_normal(
    U: np.ndarray,
    clip_eps: float = 1e-12,
) -> np.ndarray:
    """
    Map unform U in [0,1) to standard normals Z via inverse-CDF
    """
    if clip_eps is not None and clip_eps > 0:         #  To avoid infs at exactly 0 or 1 in finite precision
        U = np.clip(U, clip_eps, 1.0 - clip_eps)

    return norm.ppf(U)


def prn_normals(n: int, d: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Convenience: i.i.d. standard normals via PRN uniforms + ppf.
    """
    return u_to_normal(prn_uniform(n, d, seed=seed))


def sobol_normal_replicates(
    n: int,
    d: int,
    R: int,
    base_seed: int = 0,
    use_base2: bool = False,
    clip_eps: float = 1e-12,
) -> List[np.ndarray]:
    """
    Directly return R arrays of standard normals built from scrambled Sobol sets.
    Each element is shape (n, d).
    """
    reps_U = sobol_replicates(
        n=n, d=d, R=R, base_seed=base_seed, use_base2=use_base2,
    )
    return [u_to_normal(U, clip_eps=clip_eps) for U in reps_U]

