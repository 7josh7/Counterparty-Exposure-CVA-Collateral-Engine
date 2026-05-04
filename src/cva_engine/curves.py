"""Discount curve utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cva_engine.config import CurveConfig
from cva_engine.hull_white import hw_discount_factor


@dataclass(frozen=True)
class DiscountCurve:
    """Continuously compounded zero-rate curve with linear interpolation."""

    tenors: np.ndarray
    zero_rates: np.ndarray

    @classmethod
    def from_config(cls, config: CurveConfig) -> "DiscountCurve":
        tenors = np.asarray(config.tenors, dtype=float)
        zero_rates = np.asarray(config.zero_rates, dtype=float)
        order = np.argsort(tenors)
        return cls(tenors=tenors[order], zero_rates=zero_rates[order])

    def zero_rate(self, maturity: float | np.ndarray) -> float | np.ndarray:
        maturity_array = np.asarray(maturity, dtype=float)
        clipped = np.maximum(maturity_array, 0.0)
        rates = np.interp(
            clipped,
            self.tenors,
            self.zero_rates,
            left=self.zero_rates[0],
            right=self.zero_rates[-1],
        )
        if np.isscalar(maturity):
            return float(rates)
        return rates

    def df(self, maturity: float | np.ndarray) -> float | np.ndarray:
        maturity_array = np.asarray(maturity, dtype=float)
        dfs = np.exp(-self.zero_rate(maturity_array) * maturity_array)
        if np.isscalar(maturity):
            return float(dfs)
        return dfs

    def discount_between(
        self,
        eval_time: float,
        pay_time: float | np.ndarray,
        rate_shift: float = 0.0,
        mean_reversion: float = 0.08,
        volatility: float = 0.012,
    ) -> float | np.ndarray:
        """Discount from pay_time back to eval_time with exact Hull-White A/B dynamics."""

        return hw_discount_factor(
            self,
            eval_time,
            pay_time,
            rate_shift,
            mean_reversion,
            volatility,
        )
