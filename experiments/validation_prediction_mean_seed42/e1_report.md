# Stage E1 - Original SFTMAT + TrueToken Simple Ensemble

Existing Valid predictions only; training_runs=0; Test_runs=0.

| Task | Original | TrueToken | 0.5 Ensemble | Ensemble vs Original Gain | Oracle | Error Corr |
|---|---:|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.240074 | 0.246938 | 0.235546 | 1.886% | 0.178966 | 0.614 |
| Jarvis-Bandgap_OPT | 0.125601 | 0.122059 | 0.117511 | 6.441% | 0.0897468 | 0.807 |
| Jarvis-FormationEnergy | 0.0288919 | 0.0286698 | 0.0270538 | 6.362% | 0.0215652 | 0.882 |
| Jarvis-TotalEnergy | 0.0306145 | 0.0301749 | 0.0283271 | 7.472% | 0.0224639 | 0.864 |
| Jarvis-BulkModulusKv | 9.21484 | 9.07081 | 8.81625 | 4.325% | 7.28342 | 0.889 |
| Jarvis-ShearModulusGv | 8.68575 | 8.62245 | 8.38986 | 3.407% | 6.88706 | 0.892 |
| MP-Bandgap | 0.212996 | 0.21015 | 0.201856 | 5.230% | 0.155215 | 0.782 |
| MP-FormationEnergy | 0.0204599 | 0.0199615 | 0.0188421 | 7.907% | 0.014857 | 0.849 |
| MP-BulkModuli | 0.037895 | 0.034964 | 0.0348882 | 7.935% | 0.0280414 | 0.900 |
| MP-ShearModuli | 0.0658121 | 0.0641089 | 0.062751 | 4.651% | 0.0507614 | 0.853 |

## Decision

**SIMPLE_ENSEMBLE_GENERALIZATION_GAIN**

All 10/10 tasks have strict seed42 pairs. The ensemble beats Original on 10/10 tasks and TrueToken on 10/10. JARVIS wins are 6/6 and MP wins are 4/4. The 10-task macro gain is 5.562% (JARVIS 4.982%, MP 6.431%). 0 tasks degrade and 0 degrade by more than 1%. The locked success gate passes. E2 is not launched automatically; only a multi-seed configuration is justified.

Oracle gains on paired tasks range from 20.709% to 28.546%. ORACLE IS NOT A DEPLOYABLE MODEL.

Absolute-error Pearson correlations range from 0.614 to 0.900. Disagreement quartiles contain useful but property-dependent information, while fixed averaging already improves all ten tasks. Multi-seed confirmation is justified; a router is not currently necessary.
