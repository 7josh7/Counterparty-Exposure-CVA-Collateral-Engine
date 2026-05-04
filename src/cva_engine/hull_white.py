"""Hull-White one-factor formulas used by the CVA exposure engine."""

from __future__ import annotations

import math

import numpy as np


def b_function(mean_reversion: float, start: float, end: float) -> float | np.ndarray:
    """Mercurio/Hull-White B(start, end) = (1-exp(-a(end-start))) / a."""

    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    tau = np.maximum(end_array - start_array, 0.0)
    a = float(mean_reversion)
    if abs(a) < 1e-12:
        result = tau
    else:
        result = (1.0 - np.exp(-a * tau)) / a
    if np.isscalar(start) and np.isscalar(end):
        return float(result)
    return result


def integrated_variance(mean_reversion: float, volatility: float, start: float, end: float) -> float | np.ndarray:
    """Return integral_start^end sigma^2 B(u,end)^2 du for constant sigma."""

    start_array = np.asarray(start, dtype=float)
    end_array = np.asarray(end, dtype=float)
    tau = np.maximum(end_array - start_array, 0.0)
    a = float(mean_reversion)
    sigma = float(volatility)
    if sigma == 0.0:
        result = np.zeros_like(tau, dtype=float)
    elif abs(a) < 1e-10:
        result = sigma * sigma * tau**3 / 3.0
    else:
        result = (
            sigma
            * sigma
            / (a * a)
            * (
                tau
                - 2.0 * (1.0 - np.exp(-a * tau)) / a
                + (1.0 - np.exp(-2.0 * a * tau)) / (2.0 * a)
            )
        )
    if np.isscalar(start) and np.isscalar(end):
        return float(result)
    return result


def a_function(mean_reversion: float, volatility: float, start: float, end: float) -> float | np.ndarray:
    """Mercurio A(start, end) for dx=-a x dt + sigma dW."""

    result = np.exp(0.5 * integrated_variance(mean_reversion, volatility, start, end))
    if np.isscalar(start) and np.isscalar(end):
        return float(result)
    return result


def hw_discount_factor(
    curve,
    eval_time: float,
    pay_time: float | np.ndarray,
    factor_value: float,
    mean_reversion: float,
    volatility: float,
) -> float | np.ndarray:
    """Pathwise P(eval_time, pay_time) under the exact Hull-White bond formula."""

    pay_array = np.asarray(pay_time, dtype=float)
    eval_scalar = float(eval_time)
    valid_pay = np.maximum(pay_array, eval_scalar)
    base_forward = curve.df(valid_pay) / curve.df(eval_scalar)
    convexity_ratio = (
        a_function(mean_reversion, volatility, 0.0, eval_scalar)
        * a_function(mean_reversion, volatility, eval_scalar, valid_pay)
        / a_function(mean_reversion, volatility, 0.0, valid_pay)
    )
    stochastic_term = np.exp(
        -np.asarray(b_function(mean_reversion, eval_scalar, valid_pay), dtype=float)
        * float(factor_value)
    )
    result = base_forward * convexity_ratio * stochastic_term
    result = np.where(pay_array <= eval_scalar, 1.0, result)
    if np.isscalar(pay_time):
        return float(result)
    return result


def beta_integral_from_curve(
    projection_curve,
    mean_reversion: float,
    volatility: float,
    start: float,
    end: float,
) -> float:
    """Return integral_start^end beta(u)du from Mercurio Appendix A, eq. 48."""

    if end <= start:
        return 0.0
    ratio = a_function(mean_reversion, volatility, 0.0, end) / a_function(
        mean_reversion,
        volatility,
        0.0,
        start,
    )
    curve_ratio = projection_curve.df(start) / projection_curve.df(end)
    return float(math.log(ratio) + math.log(curve_ratio))


def ou_step_moments(mean_reversion: float, volatility: float, dt: float) -> tuple[float, float, float, float]:
    """Return decay, var(dx noise), var(integral noise), covariance."""

    a = float(mean_reversion)
    sigma = float(volatility)
    step = float(dt)
    if step <= 0.0:
        return 1.0, 0.0, 0.0, 0.0
    if sigma == 0.0:
        return math.exp(-a * step), 0.0, 0.0, 0.0
    if abs(a) < 1e-10:
        return (
            1.0,
            sigma * sigma * step,
            sigma * sigma * step**3 / 3.0,
            sigma * sigma * step * step / 2.0,
        )
    decay = math.exp(-a * step)
    variance_x = sigma * sigma * (1.0 - math.exp(-2.0 * a * step)) / (2.0 * a)
    variance_integral = integrated_variance(a, sigma, 0.0, step)
    covariance = sigma * sigma / (a * a) * (
        (1.0 - math.exp(-a * step)) - 0.5 * (1.0 - math.exp(-2.0 * a * step))
    )
    return decay, float(variance_x), float(variance_integral), float(covariance)
