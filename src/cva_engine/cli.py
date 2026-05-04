"""Command-line interface for the CVA engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from cva_engine.analytics import run_engine
from cva_engine.config import RunConfig
from cva_engine.curve_loader import market_diagnostics_from_snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the OTC rates counterparty exposure and CVA engine."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        help="Path to the JSON run configuration.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to the JSON run configuration.",
    )
    parser.add_argument(
        "--out",
        default="outputs/base_case",
        help="Directory for generated outputs.",
    )
    args = parser.parse_args(argv)

    config_path = args.config or args.config_path or "configs/base_case.json"
    config = RunConfig.from_json(config_path)
    result = run_engine(config)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result.profiles.to_csv(out_dir / "exposure_profiles.csv", index=False)
    result.portfolio.to_csv(out_dir / "portfolio.csv", index=False)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2)
    write_market_diagnostics(config, out_dir)

    _plot_exposures(result, out_dir / "exposure_profiles.png")
    _plot_cva_comparison(result.summary, out_dir / "cva_comparison.png")

    print("CVA engine run complete.")
    print(f"Outputs: {out_dir.resolve()}")
    if result.market_diagnostics:
        print(f"Market source: {result.market_diagnostics.get('source', 'curve_file')}")
    print(
        "CVA uncollateralized: "
        f"{result.summary['cva_uncollateralized']:,.0f}; "
        "collateralized: "
        f"{result.summary['cva_collateralized']:,.0f}; "
        "no-netting: "
        f"{result.summary['cva_no_netting']:,.0f}"
    )
    print(
        "WWR multiplier: "
        f"{result.summary['wrong_way_risk_multiplier']:.3f}; "
        "EPE netting: "
        f"{result.summary['epe_netting']:,.0f}"
    )
    return 0


def write_market_diagnostics(config: RunConfig, output_dir: Path) -> None:
    """Write imported market snapshot diagnostics to the CVA output directory."""

    if config.market.source != "curve_file" or config.market.snapshot_path is None:
        return

    diagnostics = market_diagnostics_from_snapshot(config.market.snapshot_path)
    with (output_dir / "market_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)


def _plot_exposures(result, path: Path) -> None:
    frame = result.profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(frame["time_years"], frame["ee_netting"], label="EE netting", linewidth=2.0)
    ax.plot(
        frame["time_years"],
        frame["ee_no_netting"],
        label="EE no netting",
        linewidth=2.0,
    )
    ax.plot(
        frame["time_years"],
        frame["ee_collateralized"],
        label="EE collateralized",
        linewidth=2.0,
    )
    pfe_cols = [name for name in frame.columns if name.startswith("pfe_")]
    for name in pfe_cols:
        if name.endswith("_netting"):
            ax.plot(frame["time_years"], frame[name], label=name, linestyle="--", alpha=0.8)
    ax.set_title("Exposure Profiles")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Exposure")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_cva_comparison(summary: dict[str, float], path: Path) -> None:
    labels = ["No netting", "Netting", "Collateral", "WWR"]
    values = [
        summary["cva_no_netting"],
        summary["cva_uncollateralized"],
        summary["cva_collateralized"],
        summary["wrong_way_risk_cva"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#7b8da8", "#3f6f8f", "#5f9f88", "#b0635b"])
    ax.set_title("CVA Comparison")
    ax.set_ylabel("CVA")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value / 1_000_000:.2f}mm",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
