# Release status

This directory is the server-side FAIR release organized against
`main_v21.tex` and `main_v21_zh.tex`.

## Corrected scope

- The current paper method is the independent Global pathway, independent
  Token pathway, and fixed sample-aligned 0.5/0.5 prediction mean.
- The old TextResidualSHFMat residual-fusion model is archived and explicitly
  excluded from the primary method description.
- Main test evidence uses training seeds 42 and 43; validation-only diagnostics
  are listed separately.

## Before public deposition

1. Replace placeholder author metadata and repository identifiers.
2. Attach exact ID-keyed split manifests for every main task.
3. Verify redistribution terms for Materials Project, JARVIS, MatBERT, and
   generated descriptions.
4. Copy or expose the final validation-selected checkpoints and per-sample
   predictions required by the repository policy.
5. Recompute the release checksum manifest after all files are finalized.
