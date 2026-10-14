# GT-PVIN Stage E3

Decision: **NO_GT_PVIN_GAIN**

## Validation results

| Task | P0 AuxConcat | Full PVIN | GT-PVIN | GT vs P0 | GT vs Full |
|---|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.25204624 | 0.24766446 | 0.23596681 | 6.380% | 4.723% |
| Jarvis-Bandgap_OPT | 0.12041502 | 0.12228178 | 0.11938400 | 0.856% | 2.370% |
| Jarvis-FormationEnergy | 0.03053209 | 0.03016251 | 0.03040583 | 0.414% | -0.807% |
| Jarvis-TotalEnergy | 0.03254130 | 0.03277423 | 0.03254645 | -0.016% | 0.695% |
| Jarvis-BulkModulusKv | 8.92406620 | 8.90202891 | 8.65939697 | 2.966% | 2.726% |
| Jarvis-ShearModulusGv | 8.74455961 | 8.30066958 | 8.57459982 | 1.944% | -3.300% |
| MP-Bandgap | 0.20608982 | 0.20347744 | 0.20272969 | 1.630% | 0.367% |
| MP-FormationEnergy | 0.02050838 | 0.02076453 | 0.02134203 | -4.065% | -2.781% |
| MP-BulkModuli | 0.03972595 | 0.03838212 | 0.04009561 | -0.931% | -4.464% |
| MP-ShearModuli | 0.07076258 | 0.07129604 | 0.07061763 | 0.205% | 0.952% |

## Required answers

1. GT-only more stable than Full PVIN: Yes.
2. P2 beats P0: 7/10.
3. 10-task macro gain vs P0: 0.938%.
4. JARVIS macro: 2.090%; MP macro: -0.790%.
5. Energetic macro changed from nan% to nan%.
6. Repaired previous Full-PVIN negative tasks: 2/4.
7. Mechanical macro: nan% (Full PVIN was nan%).
8. zero_u_gt degradation count: 10/10.
9. Parameter reduction vs Full PVIN: 24640 parameters.
10. Main single-model candidacy: Not supported as main method.
11. Worth seed43 confirmation: No, unless used as exploratory evidence.
12. NEW_TRAINING_RUNS = 10; TEST_FORWARD_COUNT = 0.
