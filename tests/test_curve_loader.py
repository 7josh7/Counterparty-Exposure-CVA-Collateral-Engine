from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cva_engine.curve_loader import market_diagnostics_from_snapshot, market_from_snapshot


class CurveLoaderTests(unittest.TestCase):
    def test_market_from_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "snapshot.json"
            snapshot_path.write_text(json.dumps(_sample_snapshot()), encoding="utf-8")

            market = market_from_snapshot(snapshot_path)

        self.assertGreater(market.discount_curve.df(1.0), 0.0)
        self.assertGreater(market.projection_curve.df(1.0), 0.0)
        self.assertGreater(market.forward_rate(1.0, 1.25), -0.05)

    def test_market_diagnostics_from_snapshot_omits_local_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_path = Path(tmp) / "snapshot.json"
            snapshot_path.write_text(json.dumps(_sample_snapshot()), encoding="utf-8")

            diagnostics = market_diagnostics_from_snapshot(snapshot_path)

        self.assertEqual(diagnostics["source"], "multi_curve_sofr")
        self.assertIn("diagnostics", diagnostics)
        self.assertNotIn("project_root", diagnostics)


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
            ],
        },
        "projection_curve": {
            "label": "sofr_projection",
            "nodes": [
                {"date": "2025-07-15", "time": 0.25, "df": 0.9878, "zero_rate": 0.0491},
                {"date": "2025-10-15", "time": 0.50, "df": 0.9758, "zero_rate": 0.0490},
                {"date": "2026-04-15", "time": 1.00, "df": 0.9522, "zero_rate": 0.0490},
                {"date": "2027-04-15", "time": 2.00, "df": 0.9076, "zero_rate": 0.0485},
            ],
        },
        "model": {"mean_reversion": 0.03, "sigma": 0.01},
        "diagnostics": {"projection_monotone": True, "discount_monotone": True},
    }


if __name__ == "__main__":
    unittest.main()
