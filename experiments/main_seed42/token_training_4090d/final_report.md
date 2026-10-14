# Final 10-Task Benchmark Report

## Original SFTMAT

Original SFTMAT is the complete graph-text `SFTGNNMultimodal` model, not a
graph-only model. It concatenates a 128-dimensional SFTGNN crystal embedding
with a 64-dimensional projection of the frozen 768-dimensional MatBERT CLS
feature and predicts with the original `192 -> 256 -> 1` property head.

## Main result

- Final status: `NO_SFTMAT_IMPROVEMENT`
- Ours wins: **4/10** tasks.
- JARVIS wins: **3/6** tasks.
- MP wins: **1/4** tasks.
- Ten-task macro relative improvement: **0.405590%**.
- JARVIS macro relative improvement: **1.282763%**.
- MP macro relative improvement: **-0.910170%**.
- Worst degradation: **Jarvis-TotalEnergy** at **-3.287629%**.
- Ours may replace Original SFTMAT as the paper main model: **No**.

## Scope and provenance

Only Ours was newly trained, once per task with seed 42. Original SFTMAT was
not retrained: its ten frozen paper results were reused. The MBJ comparison uses
the retained Original SFTMAT seed-42 row (Test MAE 0.25798455), while the
paper-reported six-seed MBJ mean (0.26355210) is retained in the baseline CSV
as a provenance note. Nine non-MBJ frozen baselines survive as aggregate
paper-table rows with recorded raw-log paths; the raw logs are not present in
this clone.

All task-level aggregation averages relative improvement rather than raw MAE.
Every checkpoint was selected by Valid only, frozen before Test, and evaluated
on Test exactly once. Test was not used to retrain, tune, remove seeds, or
select a model.
