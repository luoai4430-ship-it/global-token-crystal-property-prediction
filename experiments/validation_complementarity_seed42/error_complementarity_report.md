# Valid Error Complementarity: Original vs TrueToken

Seed 42, paired Valid samples only. No training and no Test forward.

| Task | N | TrueToken wins | Original MAE | TrueToken MAE | Mean ensemble | Oracle | Abs-error corr | Oracle gain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 1817 | 22.6% | 0.240074 | 0.246938 | 0.235546 | 0.178966 | 0.614 | 25.5% |
| Jarvis-BulkModulusKv | 1968 | 47.6% | 9.21484 | 9.07081 | 8.81625 | 7.28342 | 0.889 | 21.0% |
| MP-Bandgap | 5000 | 30.9% | 0.212996 | 0.21015 | 0.201856 | 0.155215 | 0.782 | 27.1% |
| MP-BulkModuli | 583 | 52.0% | 0.037895 | 0.034964 | 0.0348882 | 0.0280414 | 0.900 | 26.0% |

## Macro diagnostics

- truetoken_win_rate_percent: 38.2623
- average_gain_vs_original_percent: 4.8439
- oracle_gain_vs_original_percent: 24.8859
- absolute_error_correlation: 0.7958
- signed_error_correlation: 0.8059
- disagreement_advantage_correlation: -0.0875

Interpretation should distinguish oracle headroom from deployable routing: oracle uses labels and is only an upper bound.
No Test split was loaded or evaluated.
