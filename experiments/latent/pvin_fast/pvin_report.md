# PVIN Fast Screen

Decision: **STRONG_PVIN_SIGNAL**

## Validation MAE

| Task | B0 Token-only | B1 NaiveConcat | P0 AuxConcat | P1 PVIN | P1 vs B1 | P1 vs P0 | P1 vs B0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.26661300 | 0.27861283 | 0.25204624 | 0.24766446 | 11.108% | 1.738% | 7.107% |
| Jarvis-BulkModulusKv | 9.07088284 | 8.99130992 | 8.92406620 | 8.90202891 | 0.993% | 0.247% | 1.861% |
| MP-Bandgap | 0.21083623 | 0.21198018 | 0.20608982 | 0.20347744 | 4.011% | 1.268% | 3.490% |
| MP-BulkModuli | 0.04160260 | 0.04282901 | 0.03972595 | 0.03838212 | 10.383% | 3.383% | 7.741% |

- P1 wins versus B1: 4/4; macro gain vs B1: 6.624%.
- P1 wins versus P0: 4/4; macro gain vs P0: 1.659%.
- P1 wins versus rerun Token-only: 4/4; macro gain vs Token-only: 5.050%.
- JARVIS macro gain vs B1: 6.050%; MP macro gain vs B1: 7.197%.
- Full better than zero-all-interactions on all tasks: True.
- Interaction branches active: True; gradients active: True.
- Added parameters versus Token-only: 4.624% (<10%: True).

## Required answers

1. Single model truly beats NaiveConcat: Yes.
2. Auxiliary supervision itself is useful: Yes.
3. Pairwise interaction adds extra benefit over AuxConcat: Yes.
4. Most important interaction by zero-ablation macro delta: zero_u_gt (0.00804573 MAE increase when removed).
5. JARVIS and MP both benefit: Yes.
6. Worth expanding to 10 tasks: Yes.
7. Potential to replace current Global+Token ensemble: Possible, but requires the full 10-task protocol.

## Inference ablation

| Task | Full | zero u_gc | zero u_gt | zero u_ct | zero all |
|---|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.24770123 | 0.25091320 | 0.25492994 | 0.25113690 | 0.26714818 |
| Jarvis-BulkModulusKv | 8.89740124 | 8.88433148 | 8.91814727 | 8.90495690 | 8.94592783 |
| MP-Bandgap | 0.20357860 | 0.20351006 | 0.20757049 | 0.20462046 | 0.20947171 |
| MP-BulkModuli | 0.03838031 | 0.03851707 | 0.03859659 | 0.03816062 | 0.03896681 |

## Protocol integrity

- Validation-only checkpoint selection and reporting.
- P1 ablations use the selected checkpoint without retraining.
- No ensemble, prediction averaging, router, gate, FiLM, temperature search, or test evaluation.
- `test_forward_count = 0` for every run and ablation.
