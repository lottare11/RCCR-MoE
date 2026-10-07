# RCCR-MoE

Research code for **Role-based Consensus-Conflict Routing Mixture-of-Experts** in a rheumatoid arthritis study using SERS, transcriptomics and metabolomics.

This repository preserves the implementation used for the reported primary classifier. The final classifier uses six role experts, patient-specific routing, a global branch and a refinement branch. Shared/private decomposition is disabled. The separately fitted reliability response module is an auxiliary controlled-perturbation analysis, not the source of the reported primary predictions.

The cohort contains 139 participants, including 90 RA cases and 49 healthy controls. The reported five-fold ROC-AUC is 0.9933 ± 0.0120; pooled out-of-fold ROC-AUC is 0.9921. These are internal evaluation results.

## Installation

The checked research environment uses Linux, Python 3.12 and PyTorch 2.5.1. CPU execution is supported; the original experiments used a CUDA-capable environment. Install dependencies in a separate environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run without patient data

```bash
python tests/check_core.py
python scripts/run_demo.py
```

The demo generates artificial inputs and performs two folds of two training epochs on CPU. Its shapes and training budget are deliberately smaller than the study. It tests execution only. Demo metrics do not reproduce or validate the manuscript's performance.

## Reproduce the primary analysis with authorized inputs

Prepare the files described in [data/README.md](data/README.md), then run:

```bash
python scripts/run_primary.py --data-dir /path/to/authorized/data
```

The exact primary settings are in [configs/primary.json](configs/primary.json). Outputs are written under `results/raw_rebuild_snv_ladder_direct_20260916/tricor_moe/seed_1002/`. The legacy internal identifier `tricor_moe` is retained for compatibility with the saved research implementation; the manuscript name is RCCR-MoE. Do not use `src/train.py` as the primary entry point. It is retained only because analysis and baseline scripts import its metric utilities.

Every outer fold uses a separate internal training/validation split. Feature selection, grouping and scaling are fitted using only that internal training subset. Epoch selection and classification thresholds use the corresponding validation subset. Fold construction uses seed 1002.

## Additional analyses

See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the 12 comparison methods, modality subsets, cumulative structure analysis, input perturbations, reliability responses, attribution and pathway analysis. These analyses require authorized patient inputs and, for some baselines, additional upstream software. They were not re-trained as part of preparing this package.

The [reported_results](reported_results) directory contains aggregate published-in-manuscript values and fold summaries. It does not contain participant-level predictions, identifiers, matrices, checkpoints or clinical records. No full empirical reproduction is possible from this public package alone because participant data are not shared.

## Redraw supplementary figures

The aggregate CSV files are sufficient to redraw S1 and S2 without patient data:

```bash
python scripts/redraw_supplementary.py --data-dir reported_results --output-dir figures
```

S1 explicitly labels the three-modality SNV reference and the separately trained SERS-only controls. Its perturbation panel includes Raw, SNV and L2 with fold standard deviations. S2 uses observed mean fold AUC differences with paired bootstrap confidence intervals.

## Provenance and release status

`provenance/source_manifest.json` records source snapshots, release hashes and path-only edits to the historical research scripts. New wrappers provide explicit configuration and synthetic execution checks without changing the model equations or training losses. The original project's README described an earlier model and is intentionally replaced.

The associated manuscript is in preparation. `CITATION.cff.template` is a template for the authors to complete when citation details are available. `LICENSE_PENDING.txt` records that a software licence has not yet been selected. Third-party code is not vendored here; see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md).

The 2026-10-07 packaging checks preserved all model and training source files byte for byte. Python syntax and aggregate-data plotting were checked again. Earlier core and synthetic execution checks are recorded in `provenance/verification.json`; those execution checks were not rerun in the packaging environment, which does not contain PyTorch. No patient-cohort experiment was rerun.
