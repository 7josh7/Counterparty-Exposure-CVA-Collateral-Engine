# Counterparty Exposure, CVA and Collateral Engine

Python implementation of a counterparty exposure and CVA workflow for OTC interest-rate swap portfolios. The engine simulates future rates, revalues trades pathwise, builds exposure profiles, applies netting and collateral mechanics, and computes CVA/DVA-style valuation adjustments from credit curves.

The project is structured as an xVA analytics layer. It can run from an internal static curve or from an external OIS/SOFR market snapshot produced by the companion `multi_curve_sofr` project.

## Scope

The engine covers:

- vanilla fixed-vs-floating interest-rate swaps;
- single-curve and OIS/SOFR multi-curve valuation modes;
- one-factor Hull-White rate simulation;
- pathwise trade and portfolio revaluation;
- EE, PFE, EPE, MPFE, and EAD-style exposure statistics;
- netting-set exposure and no-netting trade-level exposure;
- collateral threshold, minimum transfer amount, margin frequency, and margin period of risk;
- unilateral CVA, collateralized CVA, no-netting CVA, DVA, and BVA;
- wrong-way-risk stress through path-dependent hazard rates;
- a simple funding-cost proxy.

This is a research and interview-quality prototype, not a production xVA library. The emphasis is on transparent methodology, reproducible outputs, and a clean separation between curve construction and counterparty-risk analytics.

## Architecture

The project is intentionally decoupled from the SOFR curve builder:

```text
multi_curve_sofr
    builds and validates OIS discount and SOFR projection curves
    exports outputs/curves/sofr_market_snapshot.json

cva_engine
    imports market_snapshot.json
    runs exposure, collateral, CVA, DVA, WWR, and funding analytics
```

The CVA engine does not import or execute `multi_curve_sofr` code. It consumes the exported JSON file through `cva_engine.curve_loader`.

## Repository Layout

```text
configs/
  base_case.json             Static single-curve CVA run
  sofr_snapshot_case.json    Snapshot-driven SOFR CVA run
  sofr_multi_curve_case.json Larger SOFR portfolio using the same snapshot interface

docs/
  mercurio_sofr_xva_alignment.tex

notebooks/
  counterparty_exposure_cva_math_demo.ipynb

src/cva_engine/
  analytics.py               End-to-end orchestration
  curve_loader.py            Market snapshot import
  market.py                  RatesMarket abstraction
  trades.py                  Swap valuation
  simulation.py              Hull-White factor simulation
  exposure.py                Exposure profile construction
  collateral.py              CSA mechanics
  credit.py                  Hazard curves, CVA, DVA, WWR

tests/
  Unit and integration tests
```

## Market Inputs

### Static Mode

`configs/base_case.json` contains the discount curve directly:

```json
"market": {
  "source": "static"
}
```

This mode uses the same curve for discounting and projection.

### Curve Snapshot Mode

`configs/sofr_snapshot_case.json` consumes a market snapshot:

```json
"market": {
  "source": "curve_file",
  "snapshot_path": "../multi_curve_sofr/outputs/curves/sofr_market_snapshot.json"
}
```

The snapshot contains:

- valuation date;
- OIS discount curve nodes;
- SOFR projection curve nodes;
- Hull-White model parameters;
- curve-build diagnostics such as futures and swap repricing errors.

## Methodology

The rates model uses one Hull-White factor:

```text
dx_t = -a x_t dt + sigma dW_t
```

For OIS/SOFR snapshot mode, OIS discount factors and SOFR projection factors are evolved pathwise with the same Hull-White `A(t,T)` / `B(t,T)` adjustment under deterministic SOFR-OIS basis. This follows the deterministic-basis setup in Mercurio's SOFR multi-curve framework.

At each exposure date the engine computes:

```text
E_t = max(V_t, 0)
EE(t) = average exposure at t
PFE(t) = exposure percentile at t
EPE = time average of EE(t)
MPFE = max_t PFE(t)
```

CVA is approximated on the exposure grid:

```text
CVA = (1 - R) * sum_i DF(0,t_i) * EE(t_i) * DeltaPD(t_i)
```

Collateralized and no-netting CVA are computed by replacing the exposure profile with the corresponding collateral-adjusted or trade-level exposure profile.

## Running the Engine

From `Counterparty Exposure CVA Collateral Engine/`:

```powershell
$env:PYTHONPATH = "src"
python -m cva_engine.cli configs/base_case.json --out outputs/base_case
```

To run with SOFR snapshot curves, first export the market snapshot from the companion project:

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

The larger SOFR portfolio case uses the same snapshot interface:

```powershell
python -m cva_engine.cli configs/sofr_multi_curve_case.json --out outputs/sofr_multi_curve_case
```

## Outputs

Each run writes:

- `summary.json`: headline CVA, DVA, exposure, WWR, and funding metrics;
- `portfolio.csv`: trade-level inception MTM and par-rate diagnostics;
- `exposure_profiles.csv`: EE, PFE, ENE, collateral, and netting profiles;
- `exposure_profiles.png`: exposure profile chart;
- `cva_comparison.png`: no-netting, netting, collateralized, and WWR CVA comparison;
- `market_diagnostics.json`: only for curve snapshot runs.

`market_diagnostics.json` deliberately excludes absolute local paths so outputs remain portable.

## Tests

With `pytest` installed:

```powershell
$env:PYTHONPATH = "src"
python -m pytest
```

Without `pytest`:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

The tests cover curve loading, Hull-White discounting, deterministic SOFR-OIS basis behavior, in-period SOFR accrual, swap valuation, exposure ordering, and snapshot-driven engine execution.

## Documentation

- `docs/mercurio_sofr_xva_alignment.tex` documents the mapping between the SOFR multi-curve formulas and the CVA exposure implementation.
- `notebooks/counterparty_exposure_cva_math_demo.ipynb` provides a mathematical walkthrough and reproduces the main output tables and charts.

## Limitations

- Trade schedules are represented on a year-fraction grid rather than as full date schedules.
- The rate model is one-factor Hull-White with constant volatility.
- SOFR-OIS basis is deterministic.
- Wrong-way risk is implemented as a stress mechanism, not a calibrated joint credit/rates model.
- Funding is represented by a proxy rather than a full recursive FVA framework.
- Closeout conventions, re-hypothecation, credit support disputes, and jump-to-default gap risk are not modeled.

## Possible Extensions

- Date-exact swap schedules and day-count handling in the CVA layer.
- Stochastic SOFR-OIS basis dynamics.
- Trade-level CVA allocation.
- Closeout and gap-risk modeling.
- Historical P-measure exposure risk alongside Q-measure valuation.
- Broader portfolio support, including basis swaps, caps/floors, and swaptions.
