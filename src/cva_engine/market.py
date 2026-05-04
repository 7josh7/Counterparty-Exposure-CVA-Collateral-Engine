"""Rates-market abstractions used by the exposure engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cva_engine.curves import DiscountCurve
from cva_engine.hull_white import beta_integral_from_curve, hw_discount_factor


@dataclass(frozen=True)
class RatesMarket:
    """Discount/projection curve container.

    The original CVA prototype is single-curve. This wrapper keeps that mode
    intact while allowing a SOFR projection curve to be supplied by the
    sibling multi-curve project.
    """

    discount_curve: DiscountCurve
    projection_curve: DiscountCurve | None = None
    mean_reversion: float = 0.08
    volatility: float = 0.012
    source: str = "static"
    label: str = "rates_market"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def single_curve(
        cls,
        curve: DiscountCurve,
        mean_reversion: float = 0.08,
        volatility: float = 0.012,
    ) -> "RatesMarket":
        return cls(
            discount_curve=curve,
            projection_curve=None,
            mean_reversion=mean_reversion,
            volatility=volatility,
            source="static",
            label="single_curve_market",
        )

    @property
    def is_multi_curve(self) -> bool:
        return self.projection_curve is not None

    def df(self, maturity: float | np.ndarray) -> float | np.ndarray:
        return self.discount_curve.df(maturity)

    def projection_df(self, maturity: float | np.ndarray) -> float | np.ndarray:
        curve = self.projection_curve or self.discount_curve
        return curve.df(maturity)

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        return self.discount_curve.zero_rate(maturity)

    def discount_between(
        self,
        eval_time: float,
        pay_time: float | np.ndarray,
        rate_shift: float = 0.0,
    ) -> float | np.ndarray:
        return hw_discount_factor(
            self.discount_curve,
            eval_time,
            pay_time,
            rate_shift,
            self.mean_reversion,
            self.volatility,
        )

    def projection_discount_between(
        self,
        eval_time: float,
        pay_time: float | np.ndarray,
        rate_shift: float = 0.0,
    ) -> float | np.ndarray:
        curve = self.projection_curve or self.discount_curve
        return hw_discount_factor(
            curve,
            eval_time,
            pay_time,
            rate_shift,
            self.mean_reversion,
            self.volatility,
        )

    def forward_rate_between(
        self,
        eval_time: float,
        start_time: float,
        end_time: float,
        rate_shift: float = 0.0,
        year_fraction: float | None = None,
    ) -> float:
        start = max(float(start_time), float(eval_time))
        end = float(end_time)
        tau = float(year_fraction) if year_fraction is not None else end - start
        if tau <= 1e-12:
            return 0.0
        start_df = self.projection_discount_between(eval_time, start, rate_shift)
        end_df = self.projection_discount_between(eval_time, end, rate_shift)
        return float((start_df / end_df - 1.0) / tau)

    def forward_rate(
        self,
        start: float,
        end: float,
        rate_shift: float = 0.0,
        eval_time: float = 0.0,
        year_fraction: float | None = None,
    ) -> float:
        """Simple projection forward rate from the market projection curve."""

        return self.forward_rate_between(
            eval_time=eval_time,
            start_time=start,
            end_time=end,
            rate_shift=rate_shift,
            year_fraction=year_fraction,
        )

    def compounded_sofr_accrual(
        self,
        start_time: float,
        eval_time: float,
        factor_integral: float,
    ) -> float:
        """Realized exp(integral_start^eval_time s(u)du) under deterministic basis."""

        start = float(start_time)
        end = float(eval_time)
        if end <= start + 1e-12:
            return 1.0
        projection = self.projection_curve or self.discount_curve
        deterministic_integral = beta_integral_from_curve(
            projection,
            self.mean_reversion,
            self.volatility,
            start,
            end,
        )
        return float(np.exp(float(factor_integral) + deterministic_integral))


def single_curve_market(
    curve: DiscountCurve,
    mean_reversion: float = 0.08,
    volatility: float = 0.012,
) -> RatesMarket:
    """Build a RatesMarket where discount and projection are the same curve."""

    return RatesMarket.single_curve(
        curve,
        mean_reversion=mean_reversion,
        volatility=volatility,
    )
