# GCTA interim analysis

Snapshot: 2026-09-05 15:44 UTC / 23:44 China Standard Time.

This is an interim analysis, not the final two-task decision. Two of four training runs are complete. MP-Bandgap is still training. All recorded test-forward counts are zero, and no Test prediction, Test summary, or Test-start marker was found in the experiment output directory.

## Validation performance

| Task | B1 NaiveConcat MAE (eV) | P2 GCTA MAE (eV) | Relative reduction | Evidence |
|---|---:|---:|---:|---|
| Jarvis-Bandgap_MBJ | 0.2866439255 | 0.2854286858 | 0.42395% | Frozen run summaries and sample predictions |
| MP-Bandgap | 0.22624170 | 0.22458541 | 0.7321% | Provisional log bests; epochs 75 and 77 of 100 |

Use frozen summary/prediction MAEs for the final comparison. JARVIS log minimum MAEs (B1 0.28619911 and P2 0.28541723) differ slightly from the selected-state re-evaluation. Checkpoint selection epochs are B1 80 and P2 91; neither has been reselected. Do not mix the log and final re-evaluation values into one final macro score.

## JARVIS mechanism evidence

| Measure | B1 | P2 |
|---|---:|---:|
| Mean attention entropy | 4.251325 | 4.347969 |
| Mean effective token count | 72.532907 | 80.067018 |
| Median maximum attention | 0.076774 | 0.063468 |
| Parameter count | 2,882,658 | 2,899,042 |

- P2 mean query CLS/graph norm ratio: 0.439840.
- Mean cosine between query CLS and graph terms: 0.381914.
- Wqc full-validation raw-MAE gradient L2 norm: 0.00332462. This is a validation gradient diagnostic, not a training-gradient time series.
- Mean per-sample summed L1 attention distance between B1 and P2: 0.278596. Mean absolute difference per token is 0.00217653; the original reporting code uses the latter convention. Mean total variation is 0.139298.
- Normal P2 MAE: 0.28542869 eV. Shuffled query CLS MAE: 0.29218303 eV, a 2.36639% increase. The prediction head retains the correct CLS in this control.
- Added parameters: 16,384, or 0.568364% of B1, below the prespecified 5% cap.

CLS conditioning is active and modifies token selection. Attention becomes broader on average rather than more sharply concentrated. The shuffle result supports a dependence on the matching query CLS. It does not by itself establish a robust incremental performance gain or a result on both tasks.

## Paired uncertainty analysis

The 1,817 JARVIS validation samples have exactly aligned IDs and targets. P2 wins on 443 samples, B1 on 391, and 983 are tied. Absolute-error Pearson correlation is 0.925993.

The mean paired reduction, MAE(B1) minus MAE(P2), is 0.00121524 eV. An IID paired sample bootstrap with 10,000 resamples and analysis RNG seed 20260905 gives a percentile 95% interval of [-0.0137635, 0.0155580] eV. The interval includes zero. This describes sample uncertainty conditional on these fitted models; it does not quantify training-seed uncertainty or account for possible dependence among related materials.

## Query-zero implementation correction

The original control zeroed the raw CLS before a biased projection and affine LayerNorm. This did not remove qc: the completed JARVIS checkpoint produced a query-CLS term with L2 norm 10.38643 under that control. Therefore the previously reported zero-control MAE of 0.28602829 eV is not evidence for the requested q=normalize(qg) ablation.

The inference-only fix zeroes the projected query CLS embedding immediately before the bias-free Wqc. A CPU regression check using the trained nongraph weights and synthetic graph embeddings confirmed:

- qc is exactly zero and attention equals the graph-only-query calculation.
- The normal and shuffled paths are exactly unchanged.
- The prediction head's correct CLS representation is unchanged.

Training processes and model checkpoints were not restarted or modified. The final summarization subprocess will repair validation controls sequentially after all four training summaries exist. It preserves each original summary, checkpoint, normal prediction file, and original control file; corrected controls are written to separate sidecars and used by final tables. The monitor was updated to verify this before any final decision or shutdown.

## Current interpretation

There is preliminary evidence that CLS-conditioned queries influence token selection, at a small parameter cost. JARVIS performance improves modestly, but the paired uncertainty interval does not establish a reliable advantage. MP performance and corrected controls remain pending. The prespecified GLOBAL_CONDITIONED_TOKEN_GAIN decision cannot yet be awarded, and four-task expansion has not been launched.
