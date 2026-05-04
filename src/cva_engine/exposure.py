"""Exposure profile calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cva_engine.market import RatesMarket
from cva_engine.trades import VanillaSwap


@dataclass(frozen=True)
class ExposureProfile:
    times: np.ndarray
    portfolio_values: np.ndarray
    positive_exposures: np.ndarray
    negative_exposures: np.ndarray
    ee: np.ndarray
    ene: np.ndarray
    pfe: np.ndarray
    epe: float
    mpfe: float
    ead: float


def value_portfolio_paths(
    portfolio: list[VanillaSwap],
    market: RatesMarket,
    times: np.ndarray,
    rate_shifts: np.ndarray,
    integrated_factors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return portfolio values and per-trade values for every path/date."""

    num_paths, num_times = rate_shifts.shape
    trade_values = np.zeros((num_paths, num_times, len(portfolio)), dtype=float)

    for date_index, eval_time in enumerate(times):
        shifts_at_date = rate_shifts[:, date_index]
        integrals_at_date = (
            integrated_factors[:, date_index]
            if integrated_factors is not None
            else np.zeros_like(shifts_at_date)
        )
        for trade_index, trade in enumerate(portfolio):
            values = []
            for path_index, shift in enumerate(shifts_at_date):
                if integrated_factors is None:
                    integral_at = None
                else:
                    path_integrals = integrated_factors[path_index]

                    def integral_at(target_time: float, path_integrals=path_integrals) -> float:
                        return float(np.interp(float(target_time), times, path_integrals))

                values.append(
                    trade.value(
                        float(eval_time),
                        market,
                        float(shift),
                        factor_integral=float(integrals_at_date[path_index]),
                        factor_integral_at=integral_at,
                    )
                )
            trade_values[:, date_index, trade_index] = values

    portfolio_values = np.sum(trade_values, axis=2)
    return portfolio_values, trade_values


def profile_from_values(
    times: np.ndarray,
    portfolio_values: np.ndarray,
    pfe_percentile: float,
    ead_alpha: float = 1.4,
) -> ExposureProfile:
    positive = np.maximum(portfolio_values, 0.0)
    negative = np.maximum(-portfolio_values, 0.0)
    return profile_from_exposures(
        times=times,
        portfolio_values=portfolio_values,
        positive_exposures=positive,
        negative_exposures=negative,
        pfe_percentile=pfe_percentile,
        ead_alpha=ead_alpha,
    )


def profile_from_exposures(
    times: np.ndarray,
    portfolio_values: np.ndarray,
    positive_exposures: np.ndarray,
    negative_exposures: np.ndarray,
    pfe_percentile: float,
    ead_alpha: float = 1.4,
) -> ExposureProfile:
    ee = np.mean(positive_exposures, axis=0)
    ene = np.mean(negative_exposures, axis=0)
    pfe = np.quantile(positive_exposures, pfe_percentile, axis=0)

    horizon = max(float(times[-1] - times[0]), 1e-12)
    epe = float(np.trapezoid(ee, times) / horizon)
    mpfe = float(np.max(pfe))
    ead = float(ead_alpha * epe)

    return ExposureProfile(
        times=times,
        portfolio_values=portfolio_values,
        positive_exposures=positive_exposures,
        negative_exposures=negative_exposures,
        ee=ee,
        ene=ene,
        pfe=pfe,
        epe=epe,
        mpfe=mpfe,
        ead=ead,
    )
