# GCTA Fast Screen

Decision: **WEAK_GLOBAL_CONDITIONING_SIGNAL**

## Validation MAE

| Task | B1 Naive concat | P2 GCTA |
|---|---:|---:|
| Jarvis-Bandgap_MBJ | 0.28664393 | 0.28542869 |
| MP-Bandgap | 0.21901605 | 0.21787429 |

- P2 wins versus B1: 2/2; two-task macro gain: 0.473%.
- No task degrades by more than 0.5%: True.
- Query CLS controls favor the correct CLS on both tasks: True.
- Wqc branch is active: True.
- GCTA attention changes relative to B1: True.
- Added parameters versus B1: 0.568% (<5%: True).

## Required questions

1. GCTA exceeds naive concatenation: Yes.
2. Correct query CLS beats shuffled and zero controls on both tasks: True.
3. Mean ||qc||/||qg|| by task: Jarvis-Bandgap_MBJ=0.4398, MP-Bandgap=0.1938.
4. Mean L1 distance between GCTA and B1 attention by task: Jarvis-Bandgap_MBJ=0.002177, MP-Bandgap=0.001614.
5. Worth extending to four tasks: Only as a weak-signal confirmation.

## Protocol integrity

- Validation-only checkpoint selection and reporting.
- Query controls use the selected P2 checkpoint without retraining.
- No rank, temperature, loss-weight, gate, router, or ensemble search.
- `test_forward_count = 0` for every run and control.
