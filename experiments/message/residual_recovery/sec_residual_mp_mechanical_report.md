# Residual SEC-SFT MP mechanical recovery screen

Decision: **STRONG_SEC_RESIDUAL_RECOVERY**

Validation-only seed42 screen. No test evaluation was run.

## Results

| Task | Original | Current SEC-SFT | Residual SEC-SFT | Residual vs Original | Recovery vs SEC-SFT |
|---|---:|---:|---:|---:|---:|
| MP-BulkModuli | 0.0378949879 | 0.0393518451 | 0.037176889 | 1.895% | 5.527% |
| MP-ShearModuli | 0.0658285211 | 0.0694838954 | 0.0656473727 | 0.275% | 5.521% |

## Required answers

1. Residual SEC-SFT fixes MP-BulkModuli: True.
2. Residual SEC-SFT fixes MP-ShearModuli: True.
3. Better than current SEC-SFT on 2/2: True.
4. Better than Original on 2/2: True.
5. Residual branch learned nonzero corrections: True.
6. Full better than residual-off: 2/2.
7. Reaches 4-task expansion gate: True.
8. test_forward_count = 0.
