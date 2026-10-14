# Manuscript experiment records

This directory contains existing records supporting `paper/main_v24.pdf`.
Records were collected on 2026-10-14; no training or model inference was
performed during collection. Negative results, failed attempts and recovery
logs are preserved when present in the selected source directories.

The export contains 1,164 original files: 411 logs, 433 JSON records,
248 CSV tables/prediction files, and 72 other text/script records. All
1,164 source SHA-256 hashes were verified after transfer. The seed42/43
paired test CSVs reproduce the saved per-task main MAEs within numerical
precision; this verification reads saved predictions, not model outputs.

## Evidence index

| Directory | Scope and source |
| --- | --- |
| `main_seed42/` | Server seed42 paired test predictions, Token training and Global-completion records |
| `main_seed43/` | Previously verified local backup of the original seed43 server run: validation/test summaries, paired test predictions, validation predictions, queue/recovery/preflight logs and freeze manifest |
| `matched_text_ablation/` | Server CLS/MaskedMean/CosineAttention validation study: 6 tasks x 3 variants, 18 summaries and 18 run logs; each summary contains 300 history entries; queue and recovery scripts/logs are included |
| `latent/` | TAGIF, GCTA, GCTR, PVIN and GT-PVIN validation diagnostics and available run records |
| `message/` | SEC-SFT and residual recovery/preservation records, including interrupted attempts |
| `shared_backbone/` | SBDVF six-task and extra-task records; see the baseline warning below |
| `paper_analysis/` | Existing same-view controls, complementarity, attention and literature/split provenance tables |
| `historical_references/` | Earlier benchmark and frozen-baseline snapshots, not the final main comparison |
| `manifests/` | Original absolute source paths, source type, file sizes, SHA-256 values and exclusions |

The final main Test summaries are
`../results/seed43/two_seed_task_summary.csv` and
`../results/seed43/two_seed_macro_summary.csv`, with per-seed results in
`../results/seed42/dualview_test_results.csv` and
`../results/seed43/seed43_dualview_results.csv`.
Early `results/main_test/final_10task_*` files are historical comparisons
against older frozen references; they must not replace the final paired
Global/Token comparison. Their original values have not been rewritten.

## Source and coverage limits

Server records come from `connect.westb.seetacloud.com:36397` and the exact
directories recorded in `manifests/server_export.json`. The current restored
container does not contain `/root/dualview_seed43_20260830`. Seed43 and the
TAGIF/GCTA/GCTR records therefore come from the previously saved server-audit
backups, as recorded in `manifests/local_export.json`; they are not described
as a new download from that container.

This is an export of available records, not a claim that every original
epoch log or checkpoint from every historical baseline survived. The source
benchmark report itself records absent older frozen-baseline raw logs. No
missing epoch history has been reconstructed or fabricated.

Model weights, token/graph caches, transfer archives, PID files and binary
objects are excluded. Each exclusion is listed in the export manifest.
No `.pt` checkpoint is uploaded by this evidence export. Some historical
JSON/CSV files contain original checkpoint paths/hashes; those references
are provenance, not publicly downloadable weights.

## Baseline and incomplete-run warnings

The raw SBDVF report originally used `0.02388391` for MP-FormationEnergy,
producing a reported gain of 16.659% and macro of 4.222%. That reference is
not the matched validation baseline used in the final paper. The final
manuscript uses validation MAE `0.0204598904` (displayed as `0.020460`),
gain 2.712% and balanced-subset macro 1.898%. Preserve the raw record as
history, but use the final paper and `paper_analysis/numerical_provenance_audit.csv`
for the corrected evaluation scope; do not compare a Test baseline with
validation predictions.

Residual SEC MP-Bandgap and explicitly stopped/recovery runs are process
records only, not completed formal evidence. Test results and validation
diagnostics remain separate. The matched text ablation has
`test_forward_count=0` in all 18 summaries.

Recomputing the matched normalized macros from full-precision summaries
gives 0.9979018570 (MaskedMean) and 1.0290914520 (CosineAttention).
The manuscript values derived from rounded MAEs are 0.9979018957 and
1.0290914366, respectively; both round to the same displayed values
0.9979 and 1.0291. Neither the raw records nor the manuscript values
were changed during this export.

## Verify the export

From the repository root:

```bash
python experiments/tools/verify_run_records.py
```

The verifier checks original file hashes, scans for credential markers,
checks matched-study coverage/history/Test counters, and recomputes MAE
from the already saved paired Test CSVs. It does not load a model, train,
or execute a Test forward.
