"""Collateral and netted exposure mechanics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cva_engine.config import CollateralConfig


@dataclass(frozen=True)
class CollateralResult:
    collateral_account: np.ndarray
    positive_exposures: np.ndarray
    negative_exposures: np.ndarray


def apply_collateral(
    times: np.ndarray,
    portfolio_values: np.ndarray,
    config: CollateralConfig,
) -> CollateralResult:
    """Apply a simple bilateral CSA with threshold, MTA, frequency, and MPOR.

    Positive collateral means collateral held by the bank. Negative collateral
    means collateral posted by the bank.
    """

    collateral = np.zeros_like(portfolio_values)
    margin_frequency = max(config.margin_frequency_months / 12.0, 1e-12)
    mpor = config.margin_period_of_risk_months / 12.0
    margin_dates = _margin_date_indices(times, margin_frequency)

    for date_index, eval_time in enumerate(times):
        lagged_time = max(float(eval_time) - mpor, 0.0)
        eligible = margin_dates[times[margin_dates] <= lagged_time + 1e-10]
        source_index = int(eligible[-1]) if eligible.size else 0
        target = _collateral_target(portfolio_values[:, source_index], config)
        collateral[:, date_index] = target

    held_by_bank = np.maximum(collateral, 0.0)
    posted_by_bank = np.maximum(-collateral, 0.0)
    positive_exposure = np.maximum(portfolio_values - held_by_bank, 0.0)
    negative_exposure = np.maximum(-portfolio_values - posted_by_bank, 0.0)

    return CollateralResult(
        collateral_account=collateral,
        positive_exposures=positive_exposure,
        negative_exposures=negative_exposure,
    )


def _collateral_target(values: np.ndarray, config: CollateralConfig) -> np.ndarray:
    threshold = config.threshold
    target = np.where(
        values > threshold,
        values - threshold,
        np.where(values < -threshold, values + threshold, 0.0),
    )
    return np.where(np.abs(target) >= config.minimum_transfer_amount, target, 0.0)


def _margin_date_indices(times: np.ndarray, margin_frequency: float) -> np.ndarray:
    multiples = np.round(times / margin_frequency)
    is_margin_date = np.isclose(times, multiples * margin_frequency, atol=1e-8)
    indices = np.flatnonzero(is_margin_date)
    if indices.size == 0 or indices[0] != 0:
        indices = np.insert(indices, 0, 0)
    return indices
