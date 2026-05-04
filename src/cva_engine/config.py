"""Configuration objects for the CVA engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CurveConfig:
    tenors: list[float]
    zero_rates: list[float]


@dataclass(frozen=True)
class MarketConfig:
    source: str = "static"
    snapshot_path: str | None = None


@dataclass(frozen=True)
class TradeConfig:
    trade_id: str
    notional: float
    fixed_rate: float
    maturity: float
    direction: str
    start: float = 0.0
    pay_frequency: int = 2
    day_count: str = "ACT/365F"


@dataclass(frozen=True)
class SimulationConfig:
    num_paths: int = 5000
    exposure_frequency_months: int = 3
    mean_reversion: float = 0.08
    volatility: float = 0.012
    seed: int = 42
    pfe_percentile: float = 0.95


@dataclass(frozen=True)
class CreditConfig:
    recovery_rate: float
    cds_tenors: list[float]
    cds_spreads_bps: list[float]


@dataclass(frozen=True)
class CollateralConfig:
    threshold: float = 0.0
    minimum_transfer_amount: float = 0.0
    margin_frequency_months: int = 1
    margin_period_of_risk_months: int = 0


@dataclass(frozen=True)
class WrongWayRiskConfig:
    alpha: float = 0.0


@dataclass(frozen=True)
class FundingConfig:
    funding_spread_bps: float = 0.0


@dataclass(frozen=True)
class RunConfig:
    curve: CurveConfig | None
    market: MarketConfig
    trades: list[TradeConfig]
    simulation: SimulationConfig
    counterparty_credit: CreditConfig
    own_credit: CreditConfig
    collateral: CollateralConfig
    wrong_way_risk: WrongWayRiskConfig
    funding: FundingConfig

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        _validate_raw_config(raw)
        return cls(
            curve=CurveConfig(**raw["curve"]) if "curve" in raw else None,
            market=MarketConfig(**_normalized_market(raw.get("market", {}))),
            trades=[TradeConfig(**item) for item in raw["trades"]],
            simulation=SimulationConfig(**raw.get("simulation", {})),
            counterparty_credit=CreditConfig(**raw["counterparty_credit"]),
            own_credit=CreditConfig(**raw["own_credit"]),
            collateral=CollateralConfig(**raw.get("collateral", {})),
            wrong_way_risk=WrongWayRiskConfig(**raw.get("wrong_way_risk", {})),
            funding=FundingConfig(**raw.get("funding", {})),
        )


def _validate_raw_config(raw: dict[str, Any]) -> None:
    required = ["trades", "counterparty_credit", "own_credit"]
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"Missing required config section(s): {', '.join(missing)}")

    market = _normalized_market(raw.get("market", {}))
    market_source = market.get("source", "static")
    if market_source not in {"static", "curve_file"}:
        raise ValueError("market.source must be 'static' or 'curve_file'.")

    if market_source == "static" and "curve" not in raw:
        raise ValueError("A curve section is required when market.source is 'static'.")

    if market_source == "curve_file" and not market.get("snapshot_path"):
        raise ValueError("curve_file market source requires market.snapshot_path.")

    if "curve" in raw and len(raw["curve"]["tenors"]) != len(raw["curve"]["zero_rates"]):
        raise ValueError("Curve tenors and zero_rates must have the same length.")

    if not raw["trades"]:
        raise ValueError("At least one swap trade is required.")

    for trade in raw["trades"]:
        if trade["direction"] not in {"payer", "receiver"}:
            raise ValueError("Trade direction must be 'payer' or 'receiver'.")
        if trade["maturity"] <= trade.get("start", 0.0):
            raise ValueError("Trade maturity must be greater than trade start.")

    for section in ["counterparty_credit", "own_credit"]:
        if len(raw[section]["cds_tenors"]) != len(raw[section]["cds_spreads_bps"]):
            raise ValueError(f"{section} CDS tenors and spreads must have the same length.")


def _normalized_market(raw_market: dict[str, Any]) -> dict[str, Any]:
    source = raw_market.get("source", "static")
    if source == "static_curve":
        source = "static"
    return {
        "source": source,
        "snapshot_path": raw_market.get("snapshot_path"),
    }
