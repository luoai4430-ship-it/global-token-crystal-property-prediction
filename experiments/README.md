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


## Verify the export

From the repository root:

```bash
python experiments/tools/verify_run_records.py
```

The verifier checks original file hashes, scans for credential markers,
checks matched-study coverage/history/Test counters, and recomputes MAE
from the already saved paired Test CSVs. It does not load a model, train,
or execute a Test forward.
