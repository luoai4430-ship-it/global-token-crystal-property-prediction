# Reproducibility notes for main_v24

## Main protocol

The main Global, Token, and Global+Token experiments use two independent
model-training seeds (`42`, `43`) on a common fixed partition generated with
`splitSeed123`. The split seed is a data-partition parameter and must not be
confused with either training seed.

For each task and seed, Global and Token are trained independently. The
selected checkpoint is the one with minimum validation MAE. Test predictions
are then generated once for the selected checkpoint, aligned by sample ID, and
averaged with the fixed 0.5/0.5 rule.

## Entry points

- `code/main_method/scripts/run_paper_benchmark.py`: fresh paired replication
  for seed 42 or 43; delegates to the unchanged server paired runner.
- `code/main_method/scripts/prepare_paper_inputs.py`: graph-cache preparation
  and description selection by processed split IDs.
- `code/main_method/scripts/cache_matbert_token_features.py`: frozen MatBERT
  sequence cache with float16 on-disk features and length 128.
- `code/main_method/scripts/summarize_paper_predictions.py`: sample-aligned
  fixed prediction means and two-run summaries from saved files.

The archived `run_final_10task_benchmark.py` trains only Token for seed 42
and reads a frozen Global reference. It is not a paired two-run replication
entry point. `run_seed43_benchmark.py` retains its original seed-43-only
matrix. The paper wrapper changes the allowed training seed without changing
model definitions, losses, optimizer, scheduler, or attention.

The complete command sequence is in the root README. Fresh paired replication
requires 10 tasks x 2 predictors x 2 seeds = 40 training runs. Each seed root
freezes its 20 validation-selected checkpoints before test evaluation.

## Validation-only diagnostics

Matched text representation and interaction-location studies use their own
coverage declarations in `metadata/experiment_registry.csv`. They are not
substitutes for the main two-run test protocol and must not be merged into the
main test table.

## Data handling

Use the CrysMMNet-derived snapshot and the exact processed split IDs used by
the paper. The code's MP loader selects 60000/5000/remainder for the larger MP
tasks; the 80/10/10 MP source-reference manifests are not formal split files.
Source data, checkpoints, and large caches are not included in the Git tree.
Exact replay of historical predictions needs their original caches and
checkpoints, in addition to the released code.

The paper describes cached features as BF16. The server cache generator and
reader actually use float16 arrays; BF16 is used for property-model training
autocast. This discrepancy is recorded in `docs/PAPER_V24_CODE_AUDIT.md`.
