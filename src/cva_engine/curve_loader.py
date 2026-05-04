"""Load rate-market snapshots exported by the SOFR curve project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cva_engine.curves import DiscountCurve
from cva_engine.market import RatesMarket


def load_market_snapshot(path: str | Path) -> dict[str, Any]:
    """Load exported market_snapshot.json from multi_curve_sofr."""

    snapshot_path = Path(path)
    with snapshot_path.open("r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    _validate_snapshot(snapshot)
    return snapshot


def snapshot_curve_to_discount_curve(curve_section: dict[str, Any]) -> DiscountCurve:
    """Convert snapshot curve nodes into cva_engine.curves.DiscountCurve."""

    nodes = curve_section.get("nodes", [])
    if not nodes:
        raise ValueError("Snapshot curve section must contain nodes.")
    tenors = np.asarray([float(node["time"]) for node in nodes], dtype=float)
    zero_rates = np.asarray([float(node["zero_rate"]) for node in nodes], dtype=float)
    order = np.argsort(tenors)
    return DiscountCurve(tenors=tenors[order], zero_rates=zero_rates[order])


def market_from_snapshot(
    path: str | Path,
    mean_reversion: float | None = None,
    volatility: float | None = None,
) -> RatesMarket:
    """Build RatesMarket from exported SOFR market snapshot."""

    snapshot = load_market_snapshot(path)
    discount_curve = snapshot_curve_to_discount_curve(snapshot["discount_curve"])
    projection_curve = snapshot_curve_to_discount_curve(snapshot["projection_curve"])
    model = snapshot.get("model", {})
    return RatesMarket(
        discount_curve=discount_curve,
        projection_curve=projection_curve,
        mean_reversion=float(mean_reversion if mean_reversion is not None else model.get("mean_reversion", 0.08)),
        volatility=float(volatility if volatility is not None else model.get("sigma", 0.012)),
        source=str(snapshot.get("source", "curve_file")),
        label="ois_sofr_market_snapshot",
        diagnostics=market_diagnostics_from_snapshot(path),
    )


def market_diagnostics_from_snapshot(path: str | Path) -> dict[str, Any]:
    """Extract market diagnostics for reporting into CVA output."""

    snapshot = load_market_snapshot(path)
    return {
        "source": snapshot.get("source"),
        "valuation_date": snapshot.get("valuation_date"),
        "diagnostics": snapshot.get("diagnostics", {}),
        "model": snapshot.get("model", {}),
    }


def _validate_snapshot(snapshot: dict[str, Any]) -> None:
    required = ["valuation_date", "source", "discount_curve", "projection_curve"]
    missing = [section for section in required if section not in snapshot]
    if missing:
        raise ValueError(f"Market snapshot missing required section(s): {', '.join(missing)}")
    for section_name in ["discount_curve", "projection_curve"]:
        nodes = snapshot[section_name].get("nodes", [])
        if not nodes:
            raise ValueError(f"Market snapshot {section_name} must contain nodes.")
        previous_time = -float("inf")
        for node in nodes:
            time = float(node["time"])
            if time <= previous_time:
                raise ValueError(f"Market snapshot {section_name} times must be strictly increasing.")
            if float(node["df"]) <= 0.0:
                raise ValueError(f"Market snapshot {section_name} discount factors must be positive.")
            previous_time = time
