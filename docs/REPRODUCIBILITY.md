# Reproducibility notes for main_v21

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

- `code/scripts/run_final_10task_benchmark.py`: main ten-task benchmark.
- `code/scripts/run_seed43_benchmark.py`: paired seed-43 benchmark and
  recovery-aware output generation.
- `code/scripts/summarize_seed43_dualview.py`: paired prediction checks and
  summary generation.

## Validation-only diagnostics

Matched text representation and interaction-location studies use their own
coverage declarations in `metadata/experiment_registry.csv`. They are not
substitutes for the main two-run test protocol and must not be merged into the
main test table.

## Data handling

Raw and processed datasets are kept at their canonical server locations. This
release stores manifests and checksums rather than duplicating multi-gigabyte
graph and text caches. Before public deposition, fill in the upstream dataset
licenses, exact ID manifests, and the final author metadata.
