# SEC-SFT Stage S1 and 10-task validation continuation report
Final label: MODERATE_SEC_SFT_SIGNAL
All results are seed42 validation-set results. No test evaluation was run.

## Summary
- Wins vs Original: 7/10
- 10-task macro gain vs Original: 0.174%
- JARVIS macro gain: 1.210%
- MP macro gain: -1.379%
- Electronic macro gain: 1.383%
- Energetic macro gain: 0.764%
- Mechanical macro gain: -1.175%
- TEST_FORWARD_COUNT: 0
- TEST_LOADER_CALL_COUNT: 0

## Task results
| Task | Original valid MAE | SEC-SFT valid MAE | Gain | Best epoch |
|---|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.240073875 | 0.236994913 | 1.283% | 249 |
| Jarvis-Bandgap_OPT | 0.125537502 | 0.12470927 | 0.660% | 237 |
| Jarvis-FormationEnergy | 0.0289658173 | 0.0292906103 | -1.121% | 295 |
| Jarvis-TotalEnergy | 0.0306144948 | 0.0300818806 | 1.740% | 281 |
| Jarvis-BulkModulusKv | 9.21483803 | 8.99204515 | 2.418% | 155 |
| Jarvis-ShearModulusGv | 8.68619719 | 8.48822983 | 2.279% | 286 |
| MP-Bandgap | 0.212995555 | 0.208296428 | 2.206% | 272 |
| MP-FormationEnergy | 0.0204598904 | 0.0201172671 | 1.675% | 298 |
| MP-BulkModuli | 0.0378949879 | 0.0393518451 | -3.844% | 227 |
| MP-ShearModuli | 0.0658285211 | 0.0694838954 | -5.553% | 210 |

## Interpretation
SEC-SFT improves most JARVIS electronic/mechanical tasks and the two MP electronic/formation-energy tasks, but it degrades Jarvis-FormationEnergy and the two MP modulus tasks. The expanded validation screen therefore supports a useful but property-dependent semantic edge-conditioning signal rather than a uniformly dominant replacement for the baseline.
