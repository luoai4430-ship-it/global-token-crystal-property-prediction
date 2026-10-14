# GT-PVIN overnight confirmation
SEED42 LABEL: NO_GT_PVIN_GAIN
SEED43 STATUS: NOT_RUN
FINAL SINGLE-MODEL STATUS: GT_PVIN_NOT_CONFIRMED
NEW_TRAINING_RUNS: 10
VALIDATION_FORWARD_COUNT: validation-only summaries generated
TEST_FORWARD_COUNT: 0

## Seed42 validation summary
| Task | P0 AuxConcat | Full PVIN | GT-PVIN | GT vs P0 | GT vs Full |
|---|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.25204624 | 0.24766446 | 0.23596681 | +6.380% | +4.723% |
| Jarvis-Bandgap_OPT | 0.12041502 | 0.12228178 | 0.119384 | +0.856% | +2.370% |
| Jarvis-FormationEnergy | 0.030532093 | 0.030162513 | 0.030405829 | +0.414% | -0.807% |
| Jarvis-TotalEnergy | 0.032541298 | 0.032774233 | 0.032546453 | -0.016% | +0.695% |
| Jarvis-BulkModulusKv | 8.9240662 | 8.9020289 | 8.659397 | +2.966% | +2.726% |
| Jarvis-ShearModulusGv | 8.7445596 | 8.3006696 | 8.5745998 | +1.944% | -3.300% |
| MP-Bandgap | 0.20608982 | 0.20347744 | 0.20272969 | +1.630% | +0.367% |
| MP-FormationEnergy | 0.020508379 | 0.020764528 | 0.021342029 | -4.065% | -2.781% |
| MP-BulkModuli | 0.039725946 | 0.038382116 | 0.040095608 | -0.931% | -4.464% |
| MP-ShearModuli | 0.070762582 | 0.071296039 | 0.070617632 | +0.205% | +0.952% |

## Aggregate metrics
- Wins vs P0: 7/10
- Wins vs Full PVIN: 6/10
- 10-task macro gain vs P0: 0.938%
- 10-task macro gain vs Full PVIN: 0.048%
- JARVIS macro gain vs P0: 2.090%
- MP macro gain vs P0: -0.790%
- Electronic macro gain vs P0: 2.955%
- Energetic macro gain vs P0: -1.222%
- Mechanical macro gain vs P0: 1.046%
- Max degradation vs P0: 4.065%
- zero_u_gt degradation count: 10/10; mean contribution: 2.679%; median: 1.470%

## Decision
GT-PVIN is more stable than Full PVIN on some previously negative tasks, but it does not satisfy the preregistered generalization threshold because it wins 7/10 vs P0, the 10-task macro gain is below 1%, MP macro is negative, energetic macro remains negative, and the largest degradation exceeds 2%. Therefore seed43 is not launched and no further architecture search is allowed.
