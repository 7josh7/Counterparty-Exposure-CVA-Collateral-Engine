"""Vanilla OTC rates trade definitions and valuation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from cva_engine.config import TradeConfig
from cva_engine.market import RatesMarket


@dataclass(frozen=True)
class VanillaSwap:
    trade_id: str
    notional: float
    fixed_rate: float
    maturity: float
    direction: str
    start: float = 0.0
    pay_frequency: int = 2
    day_count: str = "ACT/365F"

    @classmethod
    def from_config(cls, config: TradeConfig) -> "VanillaSwap":
        return cls(**config.__dict__)

    @property
    def accrual(self) -> float:
        return 1.0 / self.pay_frequency

    @property
    def payment_interval(self) -> float:
        return 1.0 / self.pay_frequency

    @property
    def payment_times(self) -> np.ndarray:
        first_payment = self.start + self.payment_interval
        times = np.arange(first_payment, self.maturity + 1e-10, self.payment_interval)
        return times[times > self.start + 1e-10]

    def value(
        self,
        eval_time: float,
        market: RatesMarket,
        rate_shift: float = 0.0,
        factor_integral: float = 0.0,
        factor_integral_at: Callable[[float], float] | None = None,
    ) -> float:
        """Risk-free MTM from the bank perspective.

        Payer means the bank pays fixed and receives floating.
        Receiver means the bank receives fixed and pays floating.
        """

        if eval_time >= self.maturity - 1e-10:
            return 0.0

        future_payments = self.payment_times[self.payment_times > eval_time + 1e-10]
        if future_payments.size == 0:
            return 0.0

        discount_factors = market.discount_between(eval_time, future_payments, rate_shift)
        fixed_accruals = np.array(
            [self.cashflow_accrual(max(float(payment) - self.payment_interval, self.start), float(payment)) for payment in future_payments]
        )
        fixed_leg = self.fixed_rate * float(np.sum(fixed_accruals * discount_factors))

        if market.is_multi_curve:
            floating_leg = self._multi_curve_floating_leg(
                eval_time,
                market,
                rate_shift,
                future_payments,
                discount_factors,
                factor_integral,
                factor_integral_at,
            )
            payer_value = self.notional * (floating_leg - fixed_leg)
            if self.direction == "payer":
                return payer_value
            return -payer_value

        if eval_time < self.start:
            start_df = market.discount_between(eval_time, self.start, rate_shift)
        else:
            start_df = 1.0
        maturity_df = market.discount_between(eval_time, self.maturity, rate_shift)
        floating_leg = float(start_df - maturity_df)

        payer_value = self.notional * (floating_leg - fixed_leg)
        if self.direction == "payer":
            return payer_value
        return -payer_value

    def value_paths(
        self,
        eval_time: float,
        market: RatesMarket,
        rate_shifts: np.ndarray,
        factor_integrals: np.ndarray | None = None,
        integrated_factors: np.ndarray | None = None,
        simulation_times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorized MTM for all simulated paths at one exposure date."""

        shifts = np.asarray(rate_shifts, dtype=float)
        if eval_time >= self.maturity - 1e-10:
            return np.zeros_like(shifts, dtype=float)

        future_payments = self.payment_times[self.payment_times > eval_time + 1e-10]
        if future_payments.size == 0:
            return np.zeros_like(shifts, dtype=float)

        discount_factors = np.asarray(
            market.discount_between(eval_time, future_payments, shifts),
            dtype=float,
        )
        fixed_accruals = np.array(
            [
                self.cashflow_accrual(
                    max(float(payment) - self.payment_interval, self.start),
                    float(payment),
                )
                for payment in future_payments
            ]
        )
        fixed_leg = self.fixed_rate * np.sum(discount_factors * fixed_accruals[None, :], axis=1)

        if market.is_multi_curve:
            floating_leg = self._multi_curve_floating_leg_paths(
                eval_time=eval_time,
                market=market,
                rate_shifts=shifts,
                future_payments=future_payments,
                discount_factors=discount_factors,
                factor_integrals=np.zeros_like(shifts) if factor_integrals is None else factor_integrals,
                integrated_factors=integrated_factors,
                simulation_times=simulation_times,
            )
        else:
            if eval_time < self.start:
                start_df = market.discount_between(eval_time, self.start, shifts)
            else:
                start_df = np.ones_like(shifts, dtype=float)
            maturity_df = market.discount_between(eval_time, self.maturity, shifts)
            floating_leg = np.asarray(start_df, dtype=float) - np.asarray(maturity_df, dtype=float)

        payer_value = self.notional * (floating_leg - fixed_leg)
        if self.direction == "payer":
            return payer_value
        return -payer_value

    def par_rate(
        self,
        eval_time: float,
        market: RatesMarket,
        rate_shift: float = 0.0,
        factor_integral: float = 0.0,
        factor_integral_at: Callable[[float], float] | None = None,
    ) -> float:
        if eval_time >= self.maturity - 1e-10:
            return 0.0
        future_payments = self.payment_times[self.payment_times > eval_time + 1e-10]
        if future_payments.size == 0:
            return 0.0
        discount_factors = market.discount_between(eval_time, future_payments, rate_shift)
        fixed_accruals = np.array(
            [self.cashflow_accrual(max(float(payment) - self.payment_interval, self.start), float(payment)) for payment in future_payments]
        )
        annuity = float(np.sum(fixed_accruals * discount_factors))
        if market.is_multi_curve:
            float_leg = self._multi_curve_floating_leg(
                eval_time,
                market,
                rate_shift,
                future_payments,
                discount_factors,
                factor_integral,
                factor_integral_at,
            )
            return float(float_leg / annuity)

        if eval_time < self.start:
            start_df = market.discount_between(eval_time, self.start, rate_shift)
        else:
            start_df = 1.0
        maturity_df = market.discount_between(eval_time, self.maturity, rate_shift)
        return float((start_df - maturity_df) / annuity)

    def _multi_curve_floating_leg(
        self,
        eval_time: float,
        market: RatesMarket,
        rate_shift: float,
        future_payments: np.ndarray,
        discount_factors: np.ndarray,
        factor_integral: float,
        factor_integral_at: Callable[[float], float] | None,
    ) -> float:
        floating_leg = 0.0
        for payment_time, discount_factor in zip(future_payments, discount_factors, strict=True):
            payment = float(payment_time)
            scheduled_start = max(payment - self.payment_interval, self.start)
            if scheduled_start < eval_time < payment:
                start_integral = factor_integral_at(scheduled_start) if factor_integral_at else 0.0
                realized_integral = factor_integral - start_integral
                accrued = market.compounded_sofr_accrual(
                    scheduled_start,
                    eval_time,
                    realized_integral,
                )
                sofr_df = market.projection_discount_between(eval_time, payment, rate_shift)
                floating_leg += float(discount_factor) * (accrued / float(sofr_df) - 1.0)
                continue

            accrual_start = max(scheduled_start, eval_time)
            accrual = self.cashflow_accrual(accrual_start, payment)
            forward = market.forward_rate_between(
                eval_time,
                accrual_start,
                payment,
                rate_shift,
                year_fraction=accrual,
            )
            floating_leg += accrual * forward * float(discount_factor)
        return floating_leg

    def _multi_curve_floating_leg_paths(
        self,
        eval_time: float,
        market: RatesMarket,
        rate_shifts: np.ndarray,
        future_payments: np.ndarray,
        discount_factors: np.ndarray,
        factor_integrals: np.ndarray,
        integrated_factors: np.ndarray | None,
        simulation_times: np.ndarray | None,
    ) -> np.ndarray:
        floating_leg = np.zeros_like(rate_shifts, dtype=float)
        for payment_index, payment_time in enumerate(future_payments):
            payment = float(payment_time)
            discount_factor = discount_factors[:, payment_index]
            scheduled_start = max(payment - self.payment_interval, self.start)
            if scheduled_start < eval_time < payment:
                start_integral = _path_integral_at(
                    scheduled_start,
                    integrated_factors,
                    simulation_times,
                )
                realized_integral = factor_integrals - start_integral
                accrued = market.compounded_sofr_accrual(
                    scheduled_start,
                    eval_time,
                    realized_integral,
                )
                sofr_df = market.projection_discount_between(eval_time, payment, rate_shifts)
                floating_leg += discount_factor * (accrued / sofr_df - 1.0)
                continue

            accrual_start = max(scheduled_start, eval_time)
            accrual = self.cashflow_accrual(accrual_start, payment)
            forward = market.forward_rate_between(
                eval_time,
                accrual_start,
                payment,
                rate_shifts,
                year_fraction=accrual,
            )
            floating_leg += accrual * forward * discount_factor
        return floating_leg

    def cashflow_accrual(self, start_time: float, end_time: float) -> float:
        interval = max(float(end_time) - float(start_time), 0.0)
        convention = self.day_count.upper()
        if convention in {"ACT/360", "A/360"}:
            return interval * 365.0 / 360.0
        if convention in {"ACT/365F", "ACT/365", "A/365"}:
            return interval
        raise ValueError(f"Unsupported day-count convention: {self.day_count}")


def build_portfolio(configs: list[TradeConfig]) -> list[VanillaSwap]:
    return [VanillaSwap.from_config(config) for config in configs]


def _path_integral_at(
    target_time: float,
    integrated_factors: np.ndarray | None,
    simulation_times: np.ndarray | None,
) -> np.ndarray | float:
    if integrated_factors is None or simulation_times is None:
        return 0.0
    target = float(target_time)
    times = np.asarray(simulation_times, dtype=float)
    values = np.asarray(integrated_factors, dtype=float)
    if target <= times[0]:
        return values[:, 0]
    if target >= times[-1]:
        return values[:, -1]
    upper = int(np.searchsorted(times, target, side="left"))
    if abs(times[upper] - target) < 1e-12:
        return values[:, upper]
    lower = upper - 1
    weight = (target - times[lower]) / (times[upper] - times[lower])
    return values[:, lower] * (1.0 - weight) + values[:, upper] * weight
