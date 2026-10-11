# Changelog

Notable user-facing changes to `carculator_bus`. The first entry is prepared for release and remains unpublished.

## [0.1.2] - Unreleased

### Migration to ecoinvent 3.12

- Use the coordinated `carculator_utils 1.3.7` background with premise 2.5.4 and ecoinvent 3.12 cutoff. The foreground matrix, activity index and all 57 LCIA coefficient matrices were rebuilt together. Recalculate saved LCA results; results from the previous background are not directly interchangeable.
- Rename explicit battery chemistry selections and chemistry-specific custom parameter names from `NMC-523` to `NMC-532`. The new inventory uses the actual 5:3:2 Ni:Mn:Co chemistry. Existing engineering and price assumptions carry forward; this is not an alias for the former inventory.
- Inventory exports now default to ecoinvent 3.12. Match exported suppliers against the documented compatible background. Ecoinvent 3.9/3.10 remain explicit legacy targets, but suppliers without verified older links raise an error; requesting an older version does not convert the new background automatically.
- Brightpath supplies Brightway Excel, SimaPro CSV and foreground openLCA JSON-LD exports. Select one sample before export; all selected years are retained. Brightway requires background matching, and openLCA requires provider and elementary-flow mapping before calculation. SimaPro uses Latin-1 and retains unsupported noise quantities in audit comments.
- Python support remains `>=3.12,<3.13` with NumPy `>=1.26.4,<2`. Install matching family versions and rebuild saved models/arrays when applying the corrected inputs, costs or inventory accounting described below.

### Other changes

- Review the documentation against current code, explain units and assumptions
  in plain English, and add validation bar charts with sources and explicit
  limits. Correct outdated descriptions of exports, repeated runs and scientific
  accounting; preserve dated historical results as such.
- Resolve overlapping bundled parameter records into disjoint scopes while preserving every effective static value and uncertainty distribution. Archive original records in `data/overlap_resolution_provenance.json`; regenerate old seeded arrays after this record cleanup.
- Remove import-time global warning suppression and refresh API/documentation contracts; shared sulfur-table reductions use explicit pandas axis arguments.
- Default additional CNG pump-to-tank leakage to zero beyond the delivered-fuel supplier boundary. Remove the unqualified 0.4% overlay while retaining upstream and exhaust methane; explicit measured residuals remain supported.

### Compatibility and installation

- Require the stable `carculator_utils>=1.3.7` release, including its Brightpath runtime dependency.
- Keep core model/LCIA calculations independent of Brightway projects and imports. Export writers now come through Brightpath; `excel` remains a compatibility alias and `brightway` selects the legacy stack (`bw2io<0.9`, `bw2data<4`, `bw2calc<2`).

### Model and inventory changes

- Replace the NMC-523 option with NMC-532 for the ecoinvent 3.12 background refresh. Update explicit chemistry selections and custom parameter names; engineering priors retain their previous numerical values.
- Use the matching shared premise 2.5.4 / ecoinvent 3.12 coefficient bundle; exports default to 3.12 and reject unverified older supplier links.
- Require explicit ambient temperatures when the study country has no bundled climate row; remove automatic Swiss substitution and qualify existing city proxies.
- Make repeated `set_all()` calls rebuild from retained inputs, with stable costs/energy and retained PHEV components. Preserve explicit input edits and selected sample prices; see the shared repeat-run guide.
- Bill BEVs from grid electricity consumption, including charger losses. Preserve fuel-mode costs and model-specific cost units; verify costs against completed inventory purchases. See [charging cost accounting](docs/validity.rst#charging-cost-accounting).
- Remove automatic energy-target-driven hybridization completely, including its adjustment method and target-compliance display. Preserve annual propulsion inputs across year selections; the legacy `energy_target` argument is accepted but has no effect. Verify completed city-bus and coach energy, fuel supply, emissions and LCIA across single/multiple years, and retain physical mass-compliance checks. See [scope and migration](docs/validity.rst#bus-year-selection).
- Reject cabin-temperature settings other than 20 degrees Celsius: the empirical HVAC curve does not model thermostat sensitivity. Retain scalar/monthly ambient-temperature overrides and unchanged default results; document the restriction and verify completed models and inventories. See [temperature inputs](docs/validity.rst#bus-temperature-inputs).
- Gas buses now emit the methane represented by their additional fuel-purchase allowance; previously the lost gas was absent from direct emissions. Use the shared mass balance, include both origins in impacts/exports, and document the historical loss-rate boundary; see [validation](docs/validity.rst#additional-methane-leakage).
- Make projected costs reproducible with `stochastic(n, seed=...)`, retaining factors across sample/year selections and serialization without using NumPy's global RNG. Keep deterministic static/sensitivity factors and explicit battery prices. `stochastic(1)` now also samples cost factors; regenerate old stochastic cost results.
- Display the available sample in the bus summary when sample `0` is absent, allowing selected stochastic samples to complete.
- Correct year/sample alignment in automatic component-cost projections. Multi-year sensitivity references now match static prices and costs; sampled factors remain attached to their samples across years. Preserve existing price curves, explicit battery prices and physical/inventory outputs.
- Preserve explicit generic and selected-chemistry battery prices through cost adjustment, including scoped zero and per-sample constructor inputs. Verify completed purchase and replacement costs and unchanged default pricing; see [usage](docs/usage.rst#battery-unit-costs).

### Inventory export

- Inherit Brightpath (`>=1.0.0a6,<1.1`, v1 alpha API) writers for Brightway Excel, SimaPro CSV and foreground-only openLCA JSON-LD from `carculator_utils`; document examples and return values in the [export guide](docs/inventory_export.rst).
- Default to ecoinvent 3.12 cut-off; older 3.9/3.10 targets reject suppliers without verified links. Retain one selected sample per export, every selected year and unchanged source inventories/impacts. Brightway importers still require background matching and writing.
- Document the SimaPro Latin-1 layout and identifier changes, warnings for omitted custom noise flows, and openLCA provider/elementary-flow mapping required before calculation. Remove obsolete presamples and uncertainty-export claims.

### Documentation and verification

- Clarify the retained minimum of one replacement energy battery for charger-equipped buses, separate from cycling demand. Document initial-plus-replacement inventory accounting, fractional allocation, the three-replacement cap and cost treatment; bus calculations are unchanged.

### Known limitations

- The auxiliary calibration is transferred from one Gillig bus; a held-out cycle is not an independent vehicle validation.
- Charging strategy, passenger load, HVAC and schedule constraints must match the intended service. Availability masks do not certify every schedule.
- Bus cost outputs already include passenger normalization; do not divide them by occupancy again.

See [release preparation](RELEASING.md) for coordinated versions and verification.

## [0.1.1] - 2026-10-08

### Compatibility and installation

- Require Python 3.12 (`>=3.12,<3.13`); older Python environments must be recreated.
- Use NumPy `>=1.26.4,<2` through the shared runtime.
- Require the stable `carculator_utils>=1.3.6` release, including its export extras.
- Build wheels and source distributions from centralized `pyproject.toml` metadata.
- Keep core model/LCIA use independent of Brightway; install `excel` or `brightway` extras for export. The Brightway extra targets the legacy stack (`bw2io<0.9`, `bw2data<4`, `bw2calc<2`).
- Align documentation versions with the package version and provide complete documentation-build dependencies.

### Model and inventory changes

- Add native 2025 inputs and explicit component-efficiency priors, extended consistently across model years.
- Adopt an 8.3 kW auxiliary prior for 13 m city BEVs from 2020 onward, with triangular 6.225–10.375 kW engineering uncertainty and documented single-bus calibration limits.
- Preserve explicit battery chemistry, origin, capacity and custom input choices during initialization and sizing.
- Correct CNG efficiency accounting and exclude unavailable vehicles from bounded sizing checks; mask unavailable energy and supply outputs consistently.
- Correct annualized costs and discounted replacements; bus cost outputs remain per passenger-kilometre.
- Inherit corrected shared battery/regen physics, fuel blend accounting, hot pollutant translation and export behavior.

### Documentation and verification

- Add current installation and executable 2025 quick-start examples, migration notes and a release checklist.
- Record calibration scope, measurement boundaries and numerical consistency separately from empirical validation.
- Verify built wheels and sdist-built wheels, packaged resource hashes, installed tests with export extras and offline core-only model/LCIA smoke runs.

### Known limitations

- The auxiliary calibration is transferred from one Gillig bus; a held-out cycle is not an independent vehicle validation.
- Charging strategy, passenger load, HVAC and schedule constraints must match the intended service. Availability masks do not certify every schedule.
- Bus cost outputs already include passenger normalization; do not divide them by occupancy again.

See [validation](docs/validity.rst) and [release preparation](RELEASING.md) for scope and verification instructions.
