# TAGIF Fast Screen

Decision: **NO_TAGIF_GAIN**

## Validation MAE

| Task | B0 Token-only | B1 Naive concat | P1 TAGIF |
|---|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.29646345 | 0.27375770 | 0.29873090 |
| MP-Bandgap | 0.21734947 | 0.21553056 | 0.21992560 |

- P1 wins versus B0: 0/2; two-task macro gain: -0.975%.
- P1 wins versus B1: 0/2; two-task macro gain: -5.581%.
- Correct CLS beats shuffled and zero CLS on both tasks: True.
- Orthogonal innovation is non-collapsed: True.
- Interaction branch is active: True.
- Added parameters versus B0: 4.472% (<10%: True).

## Required questions

1. TAGIF exceeds Token-only: No.
2. TAGIF exceeds naive concatenation: No.
3. JARVIS-MBJ is repaired by Global innovation: No.
4. MP-Bandgap preserves the Token advantage: No.
5. CLS provides information beyond Token under the semantic controls: True.
6. Mean ||hc_perp||/||hc|| by task: Jarvis-Bandgap_MBJ=0.9893, MP-Bandgap=0.8335.
7. Mean ||delta_h||/||b|| by task: Jarvis-Bandgap_MBJ=0.4323, MP-Bandgap=0.6948; active=True.
8. Worth extending to four tasks: No.

## Protocol integrity

- Validation-only checkpoint selection and reporting.
- Semantic controls use the selected P1 checkpoint without retraining.
- No rank, temperature, loss-weight, gate, router, or ensemble search.
- `test_forward_count = 0` for every run and control.
