# Results organization

The primary manuscript evidence is split by evaluation scope:

- `main_test/`: the two-run ten-task Global, Token, and fixed Global+Token
  results used in the main paper.
- `seed42/` and `seed43/`: machine-readable paired-run summaries and
  verification manifests.
- `diagnostics/`: validation-only interaction and representation studies. These
  are not merged into the main test table.

Large per-sample predictions and checkpoints remain in the canonical server
audit directories referenced by the corresponding manifests. Historical
TextResidualSHFMat residual-model results are not part of this release or the
current main-method result set.
