# FAIR release for the Global--Token crystal property study

This release is organized against the current `main_v21` manuscript. The
publication's primary method is a prediction-level comparison of two
independently trained graph--text pathways:

1. **Global pathway**: SFTGNN graph encoding with a projected MatBERT CLS
   representation and a scalar regression head.
2. **Token pathway**: the same SFTGNN backbone with graph-conditioned,
   cosine-normalized attention over frozen MatBERT contextual token states,
   followed by the token-level predictor described in the manuscript.
3. **Global+Token**: sample-aligned predictions combined with the fixed
   0.5/0.5 mean after independent validation-based checkpoint selection.

The main evidence is the two-run, ten-task test protocol. The matched
CLS/mean/cosine representation study and the message-, latent-, and
shared-backbone variants are validation-only diagnostics; they are not the
primary model and are not represented as uniformly repeated two-run test
experiments.

## Release layout

- `paper/`: current English and Chinese manuscripts and figures.
- `code/`: the executable data loaders, models, benchmark runners, and
  summarization scripts used by the main study.
- `results/`: machine-readable main results, two-run summaries, and diagnostic
  outputs. Large per-sample files remain in the canonical server audit roots.
- `data/`: pointers and manifests for the canonical server datasets; raw data
  are not duplicated in this release.
- `models/`: checkpoint inventories and locations for validation-selected model
  states.
- `metadata/`: experiment registry, split/training-seed distinction, schemas,
  and checksums.
- `docs/`: data card, model card, reproducibility notes, and known limitations.
- Historical TextResidualSHFMat material is excluded from this release; it is
  not the current paper method.

## Reproduction boundary

The main reproducible entry points are:

```text
code/scripts/run_final_10task_benchmark.py
code/scripts/run_seed43_benchmark.py
code/scripts/summarize_seed43_dualview.py
```

Use the exact split and cache manifests under `metadata/` before running a
benchmark. The data partition seed (`splitSeed123`) is distinct from the
model-training seeds (`42` and `43`). No test result is selected using test
labels.

## Historical model notice

`TextResidualSHFMat` is an earlier residual-fusion model and is not included in
this release. It must not be cited as the implementation of the current
Global+Token prediction-average method.
