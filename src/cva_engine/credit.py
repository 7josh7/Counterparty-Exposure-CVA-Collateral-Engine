"""Credit curves and valuation adjustments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cva_engine.config import CreditConfig
from cva_engine.curves import DiscountCurve


@dataclass(frozen=True)
class HazardCurve:
    tenors: np.ndarray
    hazard_rates: np.ndarray
    recovery_rate: float

    @classmethod
    def from_cds_config(cls, config: CreditConfig) -> "HazardCurve":
        tenors = np.asarray(config.cds_tenors, dtype=float)
        spreads = np.asarray(config.cds_spreads_bps, dtype=float) / 10000.0
        lgd = max(1.0 - config.recovery_rate, 1e-12)
        hazards = spreads / lgd
        order = np.argsort(tenors)
        return cls(
            tenors=tenors[order],
            hazard_rates=hazards[order],
            recovery_rate=config.recovery_rate,
        )

    @property
    def lgd(self) -> float:
        return 1.0 - self.recovery_rate

    def hazard_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        maturity_array = np.asarray(maturity, dtype=float)
        rates = np.interp(
            np.maximum(maturity_array, 0.0),
            self.tenors,
            self.hazard_rates,
            left=self.hazard_rates[0],
            right=self.hazard_rates[-1],
        )
        if np.isscalar(maturity):
            return float(rates)
        return rates

    def survival_probabilities(self, times: np.ndarray) -> np.ndarray:
        hazards = self.hazard_rate(times)
        cumulative = np.zeros_like(times, dtype=float)
        for i in range(1, times.size):
            dt = times[i] - times[i - 1]
            cumulative[i] = cumulative[i - 1] + 0.5 * (hazards[i] + hazards[i - 1]) * dt
        return np.exp(-cumulative)

    def marginal_default_probabilities(self, times: np.ndarray) -> np.ndarray:
        survival = self.survival_probabilities(times)
        marginal = np.zeros_like(times, dtype=float)
        marginal[1:] = survival[:-1] - survival[1:]
        return marginal


def unilateral_cva(
    times: np.ndarray,
    expected_exposure: np.ndarray,
    discount_curve: DiscountCurve,
    hazard_curve: HazardCurve,
) -> float:
    dfs = discount_curve.df(times)
    marginal_pd = hazard_curve.marginal_default_probabilities(times)
    return float(hazard_curve.lgd * np.sum(dfs * expected_exposure * marginal_pd))


def pathwise_wrong_way_cva(
    times: np.ndarray,
    exposures: np.ndarray,
    discount_curve: DiscountCurve,
    base_hazard_curve: HazardCurve,
    driver: np.ndarray,
    alpha: float,
) -> float:
    """CVA with pathwise default intensity tied to the exposure driver."""

    if alpha == 0.0:
        return unilateral_cva(
            times=times,
            expected_exposure=np.mean(exposures, axis=0),
            discount_curve=discount_curve,
            hazard_curve=base_hazard_curve,
        )

    survival = np.ones(exposures.shape[0], dtype=float)
    total = 0.0
    for i in range(1, times.size):
        dt = times[i] - times[i - 1]
        midpoint = 0.5 * (times[i] + times[i - 1])
        base_lambda = base_hazard_curve.hazard_rate(float(midpoint))
        z = _standardize(driver[:, i])
        path_lambda = base_lambda * np.exp(alpha * z - 0.5 * alpha * alpha)
        marginal_default = survival * (1.0 - np.exp(-path_lambda * dt))
        discounted_loss = (
            base_hazard_curve.lgd
            * discount_curve.df(float(times[i]))
            * exposures[:, i]
            * marginal_default
        )
        total += float(np.mean(discounted_loss))
        survival *= np.exp(-path_lambda * dt)
    return total


def _standardize(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std < 1e-12:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std
