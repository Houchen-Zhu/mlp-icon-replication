import numpy as np

from .config import ICONConfig


def generate_regime_parameters(config: ICONConfig, rng: np.random.Generator) -> np.ndarray:
    """Sample (lambda, beta) values for each regime."""
    lambda_values = rng.uniform(config.lambda_min, config.lambda_max, size=config.n_regimes_train)
    beta_values = rng.uniform(config.beta_min, config.beta_max, size=config.n_regimes_train)
    return np.stack([lambda_values, beta_values], axis=1).astype(np.float32)


def generate_smooth_positive_path(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """Generate one smooth positive selling-rate path u(t)."""
    noise = rng.normal(loc=0.0, scale=1.0, size=n_steps)
    kernel_size = 11
    kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
    smooth = np.convolve(noise, kernel, mode="same")

    # Shape [N]. Shift and scale to keep u positive.
    # 形状 [N]。平移并缩放，保证卖出速率 u 为正。
    smooth = smooth - smooth.min()
    max_value = smooth.max()
    if max_value > 1e-8:
        smooth = smooth / max_value

    amplitude = rng.uniform(0.7, 1.3)
    offset = rng.uniform(0.1, 0.3)
    return (offset + amplitude * smooth).astype(np.float32)


def compute_exponential_impact(
    u_path: np.ndarray,
    lambda_value: float,
    beta_value: float,
    dt: float,
) -> np.ndarray:
    """Compute causal exponential propagator impact Y from one u path."""
    y_path = np.zeros_like(u_path, dtype=np.float32)
    decay = np.exp(-beta_value * dt)

    for i in range(u_path.shape[0]):
        previous = y_path[i - 1] * decay if i > 0 else 0.0
        y_path[i] = previous + lambda_value * u_path[i] * dt

    return y_path


def generate_icon_data(
    n_regimes: int,
    pairs_per_regime: int,
    n_steps: int,
    seed: int,
    lambda_min: float = 0.1,
    lambda_max: float = 0.5,
    beta_min: float = 0.462,
    beta_max: float = 9.011,
    t_final: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate regimes, selling paths, and impact paths.

    Returns:
        u_paths: shape [R, P, N]
        y_paths: shape [R, P, N]
        params: shape [R, 2], columns are lambda and beta
        times: shape [N]

    返回:
        u_paths: 形状 [R, P, N]
        y_paths: 形状 [R, P, N]
        params: 形状 [R, 2]，两列分别是 lambda 和 beta
        times: 形状 [N]
    """
    rng = np.random.default_rng(seed)
    dt = t_final / n_steps
    times = np.arange(n_steps, dtype=np.float32) * dt

    lambda_values = rng.uniform(lambda_min, lambda_max, size=n_regimes)
    beta_values = rng.uniform(beta_min, beta_max, size=n_regimes)
    params = np.stack([lambda_values, beta_values], axis=1).astype(np.float32)

    u_paths = np.zeros((n_regimes, pairs_per_regime, n_steps), dtype=np.float32)
    y_paths = np.zeros((n_regimes, pairs_per_regime, n_steps), dtype=np.float32)

    for regime_index in range(n_regimes):
        lambda_value, beta_value = params[regime_index]
        for pair_index in range(pairs_per_regime):
            u_path = generate_smooth_positive_path(n_steps, rng)
            y_path = compute_exponential_impact(u_path, lambda_value, beta_value, dt)
            u_paths[regime_index, pair_index] = u_path
            y_paths[regime_index, pair_index] = y_path

    return u_paths, y_paths, params, times


def generate_from_config(
    config: ICONConfig,
    n_regimes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return generate_icon_data(
        n_regimes=n_regimes,
        pairs_per_regime=config.pairs_per_regime,
        n_steps=config.n_steps,
        seed=seed,
        lambda_min=config.lambda_min,
        lambda_max=config.lambda_max,
        beta_min=config.beta_min,
        beta_max=config.beta_max,
        t_final=config.t_final,
    )
