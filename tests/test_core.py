from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cva_engine.analytics import run_convergence, run_engine
from cva_engine.config import RunConfig
from cva_engine.curves import DiscountCurve
from cva_engine.hull_white import hw_discount_factor
from cva_engine.market import RatesMarket, single_curve_market
from cva_engine.trades import VanillaSwap


class CvaEngineTests(unittest.TestCase):
    def test_payer_swap_gains_when_rates_rise(self) -> None:
        curve = DiscountCurve(
            tenors=np.array([0.5, 1.0, 2.0, 5.0, 10.0]),
            zero_rates=np.array([0.04, 0.04, 0.04, 0.04, 0.04]),
        )
        swap = VanillaSwap(
            trade_id="test",
            notional=100_000_000,
            fixed_rate=0.04,
            maturity=5.0,
            direction="payer",
        )
        market = single_curve_market(curve)
        self.assertGreater(swap.value(0.0, market, 0.01), swap.value(0.0, market, -0.01))

    def test_hull_white_discount_matches_initial_curve_at_time_zero(self) -> None:
        curve = DiscountCurve(
            tenors=np.array([0.5, 1.0, 2.0, 5.0]),
            zero_rates=np.array([0.04, 0.041, 0.042, 0.043]),
        )
        self.assertAlmostEqual(
            hw_discount_factor(curve, 0.0, 5.0, 0.0, 0.03, 0.01),
            curve.df(5.0),
            places=12,
        )

    def test_deterministic_basis_uses_same_hw_factor_for_ois_and_sofr(self) -> None:
        discount = DiscountCurve(
            tenors=np.array([0.5, 1.0, 2.0, 5.0]),
            zero_rates=np.array([0.03, 0.03, 0.03, 0.03]),
        )
        projection = DiscountCurve(
            tenors=np.array([0.5, 1.0, 2.0, 5.0]),
            zero_rates=np.array([0.05, 0.05, 0.05, 0.05]),
        )
        market = RatesMarket(
            discount_curve=discount,
            projection_curve=projection,
            mean_reversion=0.03,
            volatility=0.01,
        )
        eval_time = 1.0
        pay_time = 5.0
        factor = 0.012
        dynamic_ratio = market.projection_discount_between(eval_time, pay_time, factor) / market.discount_between(
            eval_time,
            pay_time,
            factor,
        )
        initial_basis_ratio = (projection.df(pay_time) / projection.df(eval_time)) / (
            discount.df(pay_time) / discount.df(eval_time)
        )
        self.assertAlmostEqual(dynamic_ratio, initial_basis_ratio, places=12)

    def test_in_period_sofr_accrual_is_included(self) -> None:
        discount = DiscountCurve(
            tenors=np.array([0.5, 1.0, 2.0]),
            zero_rates=np.array([0.03, 0.03, 0.03]),
        )
        projection = DiscountCurve(
            tenors=np.array([0.5, 1.0, 2.0]),
            zero_rates=np.array([0.05, 0.05, 0.05]),
        )
        market = RatesMarket(
            discount_curve=discount,
            projection_curve=projection,
            mean_reversion=0.03,
            volatility=0.01,
        )
        accrued = market.compounded_sofr_accrual(0.0, 0.25, factor_integral=0.0)
        self.assertGreater(accrued, 1.0)

    def test_engine_cva_ordering(self) -> None:
        config = RunConfig.from_dict(_small_config())
        result = run_engine(config)
        self.assertGreaterEqual(
            result.summary["cva_no_netting"],
            result.summary["cva_uncollateralized"],
        )
        self.assertGreaterEqual(
            result.summary["cva_uncollateralized"],
            result.summary["cva_collateralized"],
        )
        self.assertGreater(result.summary["epe_netting"], 0.0)
        self.assertGreater(result.summary["wrong_way_risk_multiplier"], 0.0)
        self.assertGreater(result.summary["total_notional"], 0.0)
        self.assertGreaterEqual(result.summary["cva_uncollateralized_se"], 0.0)
        self.assertIn("cva_uncollateralized_se_bp", result.summary)

    def test_convergence_grid_reports_pathwise_standard_errors(self) -> None:
        config = RunConfig.from_dict(_small_config())
        frame = run_convergence(config, path_counts=[100, 200])
        self.assertEqual(list(frame["mc_paths"]), [100, 200, 400])
        self.assertTrue((frame["cva_uncollateralized_se_bp"] >= 0.0).all())

    def test_multi_curve_market_uses_projection_for_forward_leg(self) -> None:
        discount = DiscountCurve(
            tenors=np.array([0.5, 1.0, 2.0, 5.0]),
            zero_rates=np.array([0.03, 0.03, 0.03, 0.03]),
        )
        projection = DiscountCurve(
            tenors=np.array([0.5, 1.0, 2.0, 5.0]),
            zero_rates=np.array([0.05, 0.05, 0.05, 0.05]),
        )
        market = RatesMarket(discount_curve=discount, projection_curve=projection)
        swap = VanillaSwap(
            trade_id="multi",
            notional=100_000_000,
            fixed_rate=0.03,
            maturity=5.0,
            direction="payer",
        )
        self.assertGreater(swap.value(0.0, market), 0.0)

    def test_engine_runs_with_curve_file_market(self) -> None:
        raw = _small_config()
        raw.pop("curve")
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "snapshot.json"
            snapshot_path.write_text(json.dumps(_sample_snapshot()), encoding="utf-8")
            raw["market"] = {
                "source": "curve_file",
                "snapshot_path": str(snapshot_path),
            }
            result = run_engine(RunConfig.from_dict(raw))
        self.assertGreaterEqual(result.summary["cva_uncollateralized"], 0.0)
        self.assertGreater(result.profiles.shape[0], 0)

    def test_json_config_loader(self) -> None:
        raw = _small_config()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            config = RunConfig.from_json(path)
        self.assertEqual(len(config.trades), 2)


def _small_config() -> dict:
    return {
        "market": {"source": "static"},
        "curve": {
            "tenors": [0.5, 1.0, 2.0, 5.0, 10.0],
            "zero_rates": [0.045, 0.043, 0.040, 0.036, 0.035],
        },
        "trades": [
            {
                "trade_id": "PAY_5Y",
                "notional": 50_000_000,
                "fixed_rate": 0.036,
                "maturity": 5.0,
                "direction": "payer",
                "pay_frequency": 2,
            },
            {
                "trade_id": "REC_3Y",
                "notional": 40_000_000,
                "fixed_rate": 0.041,
                "maturity": 3.0,
                "direction": "receiver",
                "pay_frequency": 2,
            },
        ],
        "simulation": {
            "num_paths": 400,
            "exposure_frequency_months": 6,
            "mean_reversion": 0.08,
            "volatility": 0.012,
            "seed": 7,
            "pfe_percentile": 0.95,
        },
        "counterparty_credit": {
            "recovery_rate": 0.40,
            "cds_tenors": [1.0, 3.0, 5.0],
            "cds_spreads_bps": [90, 120, 150],
        },
        "own_credit": {
            "recovery_rate": 0.40,
            "cds_tenors": [1.0, 3.0, 5.0],
            "cds_spreads_bps": [60, 80, 100],
        },
        "collateral": {
            "threshold": 1_000_000,
            "minimum_transfer_amount": 100_000,
            "margin_frequency_months": 1,
            "margin_period_of_risk_months": 1,
        },
        "wrong_way_risk": {"alpha": 0.5},
        "funding": {"funding_spread_bps": 50},
    }


def _sample_snapshot() -> dict:
    return {
        "valuation_date": "2025-04-15",
        "source": "multi_curve_sofr",
        "discount_curve": {
            "label": "ois_discount",
            "nodes": [
                {"date": "2025-07-15", "time": 0.25, "df": 0.9880, "zero_rate": 0.0483},
                {"date": "2025-10-15", "time": 0.50, "df": 0.9763, "zero_rate": 0.0480},
                {"date": "2026-04-15", "time": 1.00, "df": 0.9531, "zero_rate": 0.0480},
                {"date": "2027-04-15", "time": 2.00, "df": 0.9094, "zero_rate": 0.0475},
                {"date": "2030-04-15", "time": 5.00, "df": 0.7985, "zero_rate": 0.0450},
            ],
        },
        "projection_curve": {
            "label": "sofr_projection",
            "nodes": [
                {"date": "2025-07-15", "time": 0.25, "df": 0.9878, "zero_rate": 0.0491},
                {"date": "2025-10-15", "time": 0.50, "df": 0.9758, "zero_rate": 0.0490},
                {"date": "2026-04-15", "time": 1.00, "df": 0.9522, "zero_rate": 0.0490},
                {"date": "2027-04-15", "time": 2.00, "df": 0.9076, "zero_rate": 0.0485},
                {"date": "2030-04-15", "time": 5.00, "df": 0.7945, "zero_rate": 0.0460},
            ],
        },
        "model": {"mean_reversion": 0.03, "sigma": 0.01},
        "diagnostics": {
            "projection_monotone": True,
            "discount_monotone": True,
            "positive_projection_dfs": True,
            "positive_discount_dfs": True,
        },
    }


if __name__ == "__main__":
    unittest.main()
