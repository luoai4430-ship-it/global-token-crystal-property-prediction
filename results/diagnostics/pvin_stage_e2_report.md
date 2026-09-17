# PVIN Stage E2 10-task validation-only screen

Decision: **WEAK_PVIN_10TASK_SIGNAL**

## Ten-task validation summary

| Task | P0 AuxConcat | P1 PVIN | P1 vs P0 | Full | zero all | zero u_gt |
|---|---:|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.25204624 | 0.24766446 | 1.738% | 0.24770123 | 0.26714818 | 0.25492994 |
| Jarvis-Bandgap_OPT | 0.12041502 | 0.12228178 | -1.550% | 0.12220337 | 0.13430313 | 0.12694074 |
| Jarvis-FormationEnergy | 0.03053209 | 0.03016251 | 1.210% | 0.03016063 | 0.03042508 | 0.03023240 |
| Jarvis-TotalEnergy | 0.03254130 | 0.03277423 | -0.716% | 0.03274624 | 0.03286652 | 0.03272287 |
| Jarvis-BulkModulusKv | 8.92406620 | 8.90202891 | 0.247% | 8.89740124 | 8.94592783 | 8.91814727 |
| Jarvis-ShearModulusGv | 8.74455961 | 8.30066958 | 5.076% | 8.30254544 | 8.52295088 | 8.39862492 |
| MP-Bandgap | 0.20608982 | 0.20347744 | 1.268% | 0.20357860 | 0.20947171 | 0.20757049 |
| MP-FormationEnergy | 0.02050838 | 0.02076453 | -1.249% | 0.02075823 | 0.02084565 | 0.02090773 |
| MP-BulkModuli | 0.03972595 | 0.03838212 | 3.383% | 0.03838031 | 0.03896681 | 0.03859659 |
| MP-ShearModuli | 0.07076258 | 0.07129604 | -0.754% | 0.07119120 | 0.07670487 | 0.07259102 |

## Gate metrics

- P1 wins versus P0: 6/10.
- Ten-task macro gain versus P0: 0.865%.
- JARVIS macro gain: 1.001%; MP macro gain: 0.662%.
- Electronic macro gain: 0.485%; energetic: -0.251%; mechanical: 1.988%.
- Maximum individual degradation: 1.550%.
- Full better than zero-all interactions: 10/10; mean zero-all minus full MAE: 0.03129442.
- Full better than zero-u_gt: 9/10.
- Parameter increase versus Token-only: 4.624% (<10%: True).

## Required answers

1. PVIN stably exceeds AuxConcat on ten tasks: Not by the primary gate.
2. Ten-task macro gain: 0.865%.
3. JARVIS and MP both improve: Yes.
4. Electronic, energetic, and mechanical all improve: No.
5. Clear negative transfer: Yes, at least one task degrades beyond 1%.
6. zero-all remains consistently degraded: Yes, 10/10.
7. u_gt remains the main checked interaction by mean removal delta: No (zero_all_interactions largest among checked ablations).
8. The first four-task signal generalizes to the remaining six tasks: Partially.
9. Gate to formal confirmation stage: Not reached.
10. Test forward count remains zero: Yes.

## Protocol integrity

- Six new tasks only; previous four tasks are reused from the completed PVIN fast screen.
- P0_AuxConcat and P1_PVIN only for new tasks.
- Validation-only training and selected-checkpoint inference ablations.
- No ensemble, prediction averaging, router, gate, FiLM, rank search, temperature search, or test evaluation.
