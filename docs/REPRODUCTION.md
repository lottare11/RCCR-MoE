# Reproduction guide

Run commands from the repository root. Patient inputs remain local and must follow `data/README.md`. The recommended local location for scripts without a data argument is `data/raw_rebuild_20260916/`. Most historical scripts write to fixed relative output directories. Start from an empty result tree and preserve prior results separately.

## Main classifier and structural analyses

```bash
python scripts/run_primary.py
python scripts/run_variants.py normalization
python scripts/run_variants.py structure
python scripts/run_variants.py subsets
```

`structure` cumulatively removes routing, consensus and refinement, in that order. These are not independent one-component ablations. `subsets` retrains all seven nonempty modality combinations; it is not a missing-modality test of a single trained model. New wrapper outputs use descriptive relative names. The archived incremental-analysis script retains the historical result directory names; its `names` mapping must point to the intended retrained runs before reuse.

## Comparison methods

```bash
python -m pip install -r requirements-baselines.txt
python src/run_baseline_classical_v010.py
python src/run_baseline_deep_v010.py
```

These run Elastic Net, Random Forest, XGBoost, Early MLP, Late MLP and the study's GMU, TFN, LMF and MulT adaptations. Adapted implementations are described as adaptations, not as exact official reproductions.

MOGONET requires the upstream checkout at the recorded commit:

```bash
git clone https://github.com/txWang/MOGONET.git src/modern_baselines/_official_repos/MOGONET
git -C src/modern_baselines/_official_repos/MOGONET checkout 32d9066b7fe289b3f7cdb496a5668834079c274c
python src/run_mogonet_raw_v010.py
```

The study builds one evaluation graph per held-out subject, containing that subject and the training subjects. Other held-out subjects are not included in the same graph. Its performance is specific to this inductive adaptation.

Flexynesis uses DirectPred 1.1.14. Install its dependencies in a separate environment if they conflict with the primary environment, and use the recorded source:

```bash
git clone https://github.com/BIMSBbioinfo/flexynesis.git src/modern_baselines/_official_repos/flexynesis
git -C src/modern_baselines/_official_repos/flexynesis checkout f1305e798a2867a9defff50356604269fe5581e4
python -m pip install -e src/modern_baselines/_official_repos/flexynesis
python run_flexynesis_raw_v010.py
```

DIABLO uses R, dplyr and mixOmics 6.3.2, as recorded in the study. Use the exact recorded version when reproducing historical results.

```bash
python scripts/export_diablo_inputs.py
Rscript run_diablo_v010.R
python scripts/finalize_diablo.py
python src/aggregate_baseline_stats_v010.py
```

The last command joins the primary run with all 12 baselines, checks subject/label alignment, calculates OOF AUC differences, applies 10,000 paired stratified bootstrap draws and Holm adjustment, and writes `main_comparison_interim.csv`. Do not overwrite the supplied `reported_results` tables.

## Auxiliary reliability responses

```bash
python src/calibrate_rccr_v025_20260916.py --s0-run results/raw_rebuild_snv_ladder_direct_20260916/tricor_moe/seed_1002 --out-dir results/reliability_v025
python src/rccr_mechanism_probe_v024_20260916.py --run-dir results/reliability_v025 --out-dir results/reliability_response
```

The recorded auxiliary implementation retains its historical residual reliability-calibration machinery. The primary manuscript predictions come from the CCRM/S0 run, not from this auxiliary model. RRM parameter selection uses mean training proxy loss. Backbone parameters remain fixed while training-mode dropout is active. Input-zeroing retains a modality availability mask of one. Mild-noise scores need not vary monotonically with noise level.

## Attribution and perturbations

```bash
python src/explain_s0_raw_rebuild_v017_20260916.py --run-dir results/raw_rebuild_snv_ladder_direct_20260916/tricor_moe/seed_1002 --out-dir results/explanations
python src/integrated_gradients_s0_v019_20260916.py --run-dir results/raw_rebuild_snv_ladder_direct_20260916/tricor_moe/seed_1002 --gradx-dir results/explanations --out-dir results/integrated_gradients
python src/sers_perturbation_robustness_v016_20260916.py --run-dir results/raw_rebuild_snv_ladder_direct_20260916/tricor_moe/seed_1002 --out-dir results/sers_perturbation
```

These scripts produce participant-level outputs locally. They are not intended for public upload. Contribution decomposition includes six weighted expert margins, the global branch and refinement. Highest router weight is not synonymous with the largest contribution to the final prediction.

SERS sample-permutation and circular-shift controls are implemented in `src/train_raw_rebuild_serssanity_v001_20260916.py`. They require separate SERS-only retraining and use per-sample random transformations. See `configs/sers_control.json` and `scripts/run_sers_controls.py`. Their historical architecture retains shared/private projections and must not be presented as an exact component ablation of the final classifier.

## Pathway analysis

```bash
python src/pathway_convergence_symbolfix_highconf_v008_20260916.py
```

Supply the local annotations and KEGG references listed in `data/README.md`. This is a full-cohort, label-permutation association analysis, separate from predictive cross-validation. It uses 5,000 joint label permutations and maxT adjustment over 50 candidate pathways. It does not establish a causal mechanism or direct SERS-to-gene/metabolite mapping.

## Scope of verification

Synthetic checks validate execution, training-only feature selection and algebraic margin reconstruction. The empirical classifier and all baselines were not retrained during package preparation. `provenance/verification.json` records checks actually completed. Exact numeric reproduction also depends on original inputs, their ordering, dependency versions, hardware and random-number behavior.
