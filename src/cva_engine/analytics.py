"""End-to-end CVA engine orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from cva_engine.collateral import apply_collateral
from cva_engine.config import RunConfig
from cva_engine.credit import HazardCurve, pathwise_wrong_way_cva, unilateral_cva
from cva_engine.curve_loader import market_from_snapshot
from cva_engine.curves import DiscountCurve
from cva_engine.exposure import (
    ExposureProfile,
    profile_from_exposures,
    profile_from_values,
    value_portfolio_paths,
)
from cva_engine.market import RatesMarket, single_curve_market
from cva_engine.simulation import simulate_hull_white_factor
from cva_engine.trades import build_portfolio


@dataclass(frozen=True)
class EngineResult:
    summary: dict[str, float]
    profiles: pd.DataFrame
    portfolio: pd.DataFrame
    market_diagnostics: dict[str, Any]
    netting_profile: ExposureProfile
    collateralized_profile: ExposureProfile
    no_netting_profile: ExposureProfile


DEFAULT_CONVERGENCE_PATHS = (1_000, 2_500, 5_000, 10_000, 25_000, 50_000)


def run_engine(config: RunConfig) -> EngineResult:
    market = build_market_from_config(config)
    discount_curve = market.discount_curve
    portfolio = build_portfolio(config.trades)
    total_notional = float(sum(abs(trade.notional) for trade in portfolio))
    max_maturity = max(trade.maturity for trade in portfolio)
    simulation = simulate_hull_white_factor(max_maturity, config.simulation)

    portfolio_values, trade_values = value_portfolio_paths(
        portfolio=portfolio,
        market=market,
        times=simulation.times,
        rate_shifts=simulation.rate_shifts,
        integrated_factors=simulation.integrated_factors,
    )

    percentile = config.simulation.pfe_percentile
    netting_profile = profile_from_values(simulation.times, portfolio_values, percentile)

    no_netting_positive = np.sum(np.maximum(trade_values, 0.0), axis=2)
    no_netting_negative = np.sum(np.maximum(-trade_values, 0.0), axis=2)
    no_netting_profile = profile_from_exposures(
        times=simulation.times,
        portfolio_values=portfolio_values,
        positive_exposures=no_netting_positive,
        negative_exposures=no_netting_negative,
        pfe_percentile=percentile,
    )

    collateral = apply_collateral(simulation.times, portfolio_values, config.collateral)
    collateralized_profile = profile_from_exposures(
        times=simulation.times,
        portfolio_values=portfolio_values,
        positive_exposures=collateral.positive_exposures,
        negative_exposures=collateral.negative_exposures,
        pfe_percentile=percentile,
    )

    counterparty_hazard = HazardCurve.from_cds_config(config.counterparty_credit)
    own_hazard = HazardCurve.from_cds_config(config.own_credit)

    cva_uncollateralized = unilateral_cva(
        simulation.times,
        netting_profile.ee,
        discount_curve,
        counterparty_hazard,
    )
    cva_collateralized = unilateral_cva(
        simulation.times,
        collateralized_profile.ee,
        discount_curve,
        counterparty_hazard,
    )
    cva_no_netting = unilateral_cva(
        simulation.times,
        no_netting_profile.ee,
        discount_curve,
        counterparty_hazard,
    )
    dva_uncollateralized = unilateral_cva(
        simulation.times,
        netting_profile.ene,
        discount_curve,
        own_hazard,
    )
    dva_collateralized = unilateral_cva(
        simulation.times,
        collateralized_profile.ene,
        discount_curve,
        own_hazard,
    )

    wwr_cva = pathwise_wrong_way_cva(
        times=simulation.times,
        exposures=netting_profile.positive_exposures,
        discount_curve=discount_curve,
        base_hazard_curve=counterparty_hazard,
        driver=portfolio_values,
        alpha=config.wrong_way_risk.alpha,
    )
    funding_proxy = funding_cost_proxy(
        times=simulation.times,
        funding_need=np.maximum(-portfolio_values, 0.0)
        + np.maximum(-collateral.collateral_account, 0.0),
        discount_curve=discount_curve,
        funding_spread=config.funding.funding_spread_bps / 10000.0,
    )
    cva_uncollateralized_se = cva_standard_error(
        simulation.times,
        netting_profile.positive_exposures,
        discount_curve,
        counterparty_hazard,
    )
    cva_collateralized_se = cva_standard_error(
        simulation.times,
        collateralized_profile.positive_exposures,
        discount_curve,
        counterparty_hazard,
    )
    cva_no_netting_se = cva_standard_error(
        simulation.times,
        no_netting_profile.positive_exposures,
        discount_curve,
        counterparty_hazard,
    )

    summary = {
        "mc_paths": float(config.simulation.num_paths),
        "pfe_percentile": float(config.simulation.pfe_percentile),
        "total_notional": total_notional,
        "current_portfolio_mtm": float(np.mean(portfolio_values[:, 0])),
        "current_exposure": float(netting_profile.ee[0]),
        "epe_netting": netting_profile.epe,
        "mpfe_netting": netting_profile.mpfe,
        "ead_netting_alpha_1_4": netting_profile.ead,
        "epe_no_netting": no_netting_profile.epe,
        "mpfe_no_netting": no_netting_profile.mpfe,
        "epe_collateralized": collateralized_profile.epe,
        "mpfe_collateralized": collateralized_profile.mpfe,
        "cva_uncollateralized": cva_uncollateralized,
        "cva_collateralized": cva_collateralized,
        "cva_no_netting": cva_no_netting,
        "cva_uncollateralized_bp": bp_of_notional(cva_uncollateralized, total_notional),
        "cva_collateralized_bp": bp_of_notional(cva_collateralized, total_notional),
        "cva_no_netting_bp": bp_of_notional(cva_no_netting, total_notional),
        "cva_uncollateralized_se": cva_uncollateralized_se,
        "cva_collateralized_se": cva_collateralized_se,
        "cva_no_netting_se": cva_no_netting_se,
        "cva_uncollateralized_se_bp": bp_of_notional(
            cva_uncollateralized_se,
            total_notional,
        ),
        "cva_collateralized_se_bp": bp_of_notional(
            cva_collateralized_se,
            total_notional,
        ),
        "cva_no_netting_se_bp": bp_of_notional(cva_no_netting_se, total_notional),
        "dva_uncollateralized": dva_uncollateralized,
        "dva_collateralized": dva_collateralized,
        "bva_uncollateralized_dva_minus_cva": dva_uncollateralized - cva_uncollateralized,
        "wrong_way_risk_cva": wwr_cva,
        "wrong_way_risk_multiplier": wwr_cva / cva_uncollateralized
        if cva_uncollateralized > 0.0
        else float("nan"),
        "funding_cost_proxy": funding_proxy,
    }

    profiles = build_profiles_dataframe(
        times=simulation.times,
        netting=netting_profile,
        no_netting=no_netting_profile,
        collateralized=collateralized_profile,
        collateral_account=collateral.collateral_account,
        percentile=percentile,
    )
    portfolio_frame = build_portfolio_dataframe(portfolio, market)

    return EngineResult(
        summary=summary,
        profiles=profiles,
        portfolio=portfolio_frame,
        market_diagnostics=market.diagnostics if config.market.source == "curve_file" else {},
        netting_profile=netting_profile,
        collateralized_profile=collateralized_profile,
        no_netting_profile=no_netting_profile,
    )


def run_convergence(
    config: RunConfig,
    path_counts: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Run path-count convergence for CVA estimates and standard errors."""

    counts = sorted(set(path_counts or DEFAULT_CONVERGENCE_PATHS))
    max_paths = int(config.simulation.num_paths)
    counts = [count for count in counts if count <= max_paths]
    if max_paths not in counts:
        counts.append(max_paths)

    rows = []
    for count in counts:
        simulation = replace(config.simulation, num_paths=int(count))
        result = run_engine(replace(config, simulation=simulation))
        summary = result.summary
        rows.append(
            {
                "mc_paths": int(count),
                "cva_uncollateralized": summary["cva_uncollateralized"],
                "cva_uncollateralized_bp": summary["cva_uncollateralized_bp"],
                "cva_uncollateralized_se": summary["cva_uncollateralized_se"],
                "cva_uncollateralized_se_bp": summary["cva_uncollateralized_se_bp"],
                "cva_collateralized": summary["cva_collateralized"],
                "cva_collateralized_bp": summary["cva_collateralized_bp"],
                "cva_collateralized_se": summary["cva_collateralized_se"],
                "cva_collateralized_se_bp": summary["cva_collateralized_se_bp"],
                "mpfe_netting": summary["mpfe_netting"],
                "mpfe_collateralized": summary["mpfe_collateralized"],
            }
        )
    return pd.DataFrame(rows)


def build_market_from_config(config: RunConfig) -> RatesMarket:
    """Build RatesMarket from either static curve config or market snapshot file."""

    if config.market.source == "static":
        if config.curve is None:
            raise ValueError("Static market source requires curve config.")
        curve = DiscountCurve.from_config(config.curve)
        return single_curve_market(
            curve,
            mean_reversion=config.simulation.mean_reversion,
            volatility=config.simulation.volatility,
        )

    if config.market.source == "curve_file":
        if config.market.snapshot_path is None:
            raise ValueError("curve_file market source requires snapshot_path.")
        return market_from_snapshot(
            config.market.snapshot_path,
            mean_reversion=config.simulation.mean_reversion,
            volatility=config.simulation.volatility,
        )

    raise ValueError(f"Unsupported market source: {config.market.source}")


def cva_pathwise_contributions(
    times: np.ndarray,
    exposures: np.ndarray,
    discount_curve: DiscountCurve,
    hazard_curve: HazardCurve,
) -> np.ndarray:
    """Return independent pathwise CVA losses for MC error estimation."""

    dfs = discount_curve.df(times)
    marginal_pd = hazard_curve.marginal_default_probabilities(times)
    weights = hazard_curve.lgd * dfs * marginal_pd
    return np.sum(exposures * weights, axis=1)


def cva_standard_error(
    times: np.ndarray,
    exposures: np.ndarray,
    discount_curve: DiscountCurve,
    hazard_curve: HazardCurve,
) -> float:
    """Monte Carlo standard error for the unilateral CVA estimator."""

    contributions = cva_pathwise_contributions(
        times=times,
        exposures=exposures,
        discount_curve=discount_curve,
        hazard_curve=hazard_curve,
    )
    if contributions.size <= 1:
        return 0.0
    return float(np.std(contributions, ddof=1) / np.sqrt(contributions.size))


def bp_of_notional(value: float, total_notional: float) -> float:
    if total_notional <= 0.0:
        return float("nan")
    return float(value / total_notional * 10000.0)


def funding_cost_proxy(
    times: np.ndarray,
    funding_need: np.ndarray,
    discount_curve: DiscountCurve,
    funding_spread: float,
) -> float:
    if funding_spread == 0.0:
        return 0.0
    expected_need = np.mean(funding_need, axis=0)
    total = 0.0
    for i in range(1, times.size):
        dt = times[i] - times[i - 1]
        total += (
            discount_curve.df(float(times[i]))
            * funding_spread
            * expected_need[i]
            * dt
        )
    return float(total)


def build_profiles_dataframe(
    times: np.ndarray,
    netting: ExposureProfile,
    no_netting: ExposureProfile,
    collateralized: ExposureProfile,
    collateral_account: np.ndarray,
    percentile: float,
) -> pd.DataFrame:
    pfe_label = f"pfe_{int(round(percentile * 1000)) / 10:g}"
    return pd.DataFrame(
        {
            "time_years": times,
            "ee_netting": netting.ee,
            pfe_label + "_netting": netting.pfe,
            "ene_netting": netting.ene,
            "ee_no_netting": no_netting.ee,
            pfe_label + "_no_netting": no_netting.pfe,
            "ee_collateralized": collateralized.ee,
            pfe_label + "_collateralized": collateralized.pfe,
            "ene_collateralized": collateralized.ene,
            "expected_collateral_held_positive": np.mean(
                np.maximum(collateral_account, 0.0),
                axis=0,
            ),
            "expected_collateral_posted_positive": np.mean(
                np.maximum(-collateral_account, 0.0),
                axis=0,
            ),
        }
    )


def build_portfolio_dataframe(portfolio, market: RatesMarket) -> pd.DataFrame:
    rows = []
    for trade in portfolio:
        rows.append(
            {
                "trade_id": trade.trade_id,
                "direction": trade.direction,
                "notional": trade.notional,
                "fixed_rate": trade.fixed_rate,
                "maturity": trade.maturity,
                "pay_frequency": trade.pay_frequency,
                "market_source": market.source,
                "par_rate_at_inception": trade.par_rate(0.0, market),
                "mtm_at_inception": trade.value(0.0, market),
            }
        )
    return pd.DataFrame(rows)
