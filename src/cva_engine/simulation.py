"""Interest-rate path simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cva_engine.config import SimulationConfig
from cva_engine.hull_white import b_function, ou_step_moments


@dataclass(frozen=True)
class RateSimulationResult:
    times: np.ndarray
    rate_shifts: np.ndarray
    integrated_factors: np.ndarray


def make_exposure_grid(max_maturity: float, frequency_months: int) -> np.ndarray:
    dt = frequency_months / 12.0
    times = np.arange(0.0, max_maturity + 1e-10, dt)
    if times[-1] < max_maturity - 1e-10:
        times = np.append(times, max_maturity)
    return times


def simulate_hull_white_factor(
    max_maturity: float,
    config: SimulationConfig,
) -> RateSimulationResult:
    """Simulate a mean-reverting Gaussian level factor.

    The factor follows dx = -a x dt + sigma dW. The simulation also carries
    the realized integral of x, which is needed for in-period compounded SOFR
    accruals when an exposure date falls between coupon dates.
    """

    times = make_exposure_grid(max_maturity, config.exposure_frequency_months)
    rng = np.random.default_rng(config.seed)
    shifts = np.zeros((config.num_paths, times.size), dtype=float)
    integrated = np.zeros((config.num_paths, times.size), dtype=float)

    a = config.mean_reversion
    sigma = config.volatility
    for i in range(1, times.size):
        dt = times[i] - times[i - 1]
        decay, variance_x, variance_integral, covariance = ou_step_moments(a, sigma, dt)
        mean_integral = shifts[:, i - 1] * b_function(a, 0.0, dt)

        if variance_x <= 0.0 or variance_integral <= 0.0:
            x_noise = np.zeros(config.num_paths)
            integral_noise = np.zeros(config.num_paths)
        else:
            covariance_matrix = np.array(
                [[variance_x, covariance], [covariance, variance_integral]],
                dtype=float,
            )
            noises = rng.multivariate_normal(
                mean=np.array([0.0, 0.0]),
                cov=covariance_matrix,
                size=config.num_paths,
            )
            x_noise = noises[:, 0]
            integral_noise = noises[:, 1]
        shifts[:, i] = shifts[:, i - 1] * decay + x_noise
        integrated[:, i] = integrated[:, i - 1] + mean_integral + integral_noise

    return RateSimulationResult(times=times, rate_shifts=shifts, integrated_factors=integrated)
