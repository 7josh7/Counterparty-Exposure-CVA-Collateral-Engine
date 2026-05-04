# Counterparty Exposure, CVA and Collateral Simulation Engine

This is a compact research-style xVA prototype for OTC rates portfolios. It is deliberately not a full industrial xVA stack; it is built to answer:

> How do exposure, collateral, netting, credit spreads, funding assumptions, and wrong-way risk affect CVA for a small OTC derivatives portfolio?

The project maps directly to the Brigo counterparty-risk framing you provided: exposure simulation is separated from default simulation, PFE is computed as a future exposure percentile, EE/EPE summarize expected exposure, and CVA/DVA are calculated from exposure profiles and marginal default probabilities.

## What It Implements

- OIS/SOFR-style discount curve assumptions, or OIS/SOFR market snapshots exported by the sibling `multi_curve_sofr` project.
- Exact one-factor Hull-White `A(t,T)` / `B(t,T)` pathwise discounting for exposure revaluation.
- Deterministic SOFR-OIS basis dynamics: the same Hull-White factor drives OIS discount and SOFR projection curves.
- A 3-5 trade vanilla interest-rate swap portfolio with payer and receiver directions.
- One-factor Gaussian Hull-White-style rate simulation.
- Pathwise portfolio revaluation at future monthly or quarterly dates.
- Netting-set exposure versus no-netting trade-level exposure.
- EE, PFE, EPE, MPFE, and a simple Basel-style EAD proxy.
- Uncollateralized CVA, collateralized CVA, no-netting CVA, DVA, and BVA.
- CSA mechanics: threshold, minimum transfer amount, margin frequency, and margin period of risk.
- Wrong-way-risk stress with path-dependent hazard rates.
- A simple funding cost proxy for liquidity/funding comparison.

## Project Layout

```text
configs/base_case.json       Example portfolio, curve, credit, CSA, and simulation inputs
configs/sofr_snapshot_case.json
                             CVA run consuming ../multi_curve_sofr/outputs/curves/sofr_market_snapshot.json
configs/sofr_multi_curve_case.json
                             Larger snapshot-driven SOFR portfolio run
docs/mercurio_sofr_xva_alignment.tex
                             LaTeX note mapping Mercurio's formulas to the implementation
notebooks/                   Mathematical walkthrough notebook
src/cva_engine/              Core package
tests/                       Unit tests for valuation, exposure, collateral, and CVA behavior
outputs/                     Generated CSV, JSON, and chart artifacts
```

## Notebook Walkthrough

Open:

```text
notebooks/counterparty_exposure_cva_math_demo.ipynb
```

The notebook explains the math behind Mercurio/Hull-White curve discounting, SOFR swap valuation with in-period accrual, EE/PFE/EPE/MPFE, netting, collateral, CVA/DVA/BVA, and wrong-way risk. It also runs the base case and displays the resulting charts and tables.

## Run The Demo

From this folder:

```powershell
$env:PYTHONPATH = "src"
python -m cva_engine.cli configs/base_case.json --out outputs/base_case
```

The CVA engine can run in two modes:

1. Static single-curve mode: uses curve data directly in the config file.
2. Curve snapshot mode: consumes an exported OIS/SOFR market snapshot from the `multi_curve_sofr` project. The CVA engine does not import or execute the SOFR curve builder.

First export the SOFR snapshot from the sibling project:

```powershell
cd ..\multi_curve_sofr
python -m src.cli export-snapshot --output outputs/curves/sofr_market_snapshot.json
cd "..\Counterparty Exposure CVA Collateral Engine"
```

Then run the snapshot-based CVA case:

```powershell
$env:PYTHONPATH = "src"
python -m cva_engine.cli configs/sofr_snapshot_case.json --out outputs/sofr_snapshot_case
```

This loads the exported JSON snapshot, converts the OIS discount and SOFR projection nodes into the CVA engine's curve representation, and values the swap portfolio with separate discount and projection curves.

The run writes:

- `summary.json`
- `market_diagnostics.json`
- `exposure_profiles.csv`
- `portfolio.csv`
- `exposure_profiles.png`
- `cva_comparison.png`

## Run Tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest
```

Or, without installing `pytest`:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

If you want to install it as a local package:

```powershell
python -m pip install -e .
cva-engine --config configs/base_case.json --out outputs/base_case
```

## Modeling Notes

The swap valuation uses residual fixed-leg annuities and floating-leg discount-bond formulas. In SOFR multi-curve mode, OIS discount factors and SOFR projection discount factors are both evolved with Mercurio's exact one-factor Hull-White `A(t,T)` / `B(t,T)` bond formula under deterministic SOFR-OIS basis. Exposure simulation carries both `x_t` and the realized integral of `x_t`, so the current SOFR coupon includes already accrued compounded SOFR when an exposure date falls between coupon dates.

The CVA layer is still intentionally compact: it consumes a validated curve snapshot rather than becoming a full date-exact production rates library. Snapshot diagnostics are copied to `market_diagnostics.json` without absolute local paths, so the dependency between projects is data-based and portable.

The funding metric is a transparent proxy, not a full recursive FVA/BSDE implementation. It estimates the discounted cost of funding negative MTM and collateral posting requirements at an assumed funding spread.

## Research Extensions

Natural next upgrades:

- Calibrate the OIS curve from market instruments.
- Add date-exact swap schedules directly in the CVA layer.
- Add stochastic SOFR-OIS basis dynamics beyond Mercurio's deterministic-basis setup.
- Add closeout conventions, re-hypothecation losses, and gap-risk jumps.
- Allocate CVA back to trades by marginal or incremental contribution.
- Add historical P-measure exposure risk and Q-measure pricing side by side.
