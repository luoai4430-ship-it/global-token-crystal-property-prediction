# GCTR Fast Screen

Decision: **NO_GCTR_GAIN**

## Validation MAE

| Task | B1 Naive concat | P3 GCTR | Relative gain vs B1 |
|---|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.28735093 | 0.28952800 | -0.758% |
| MP-Bandgap | 0.22147668 | 0.21741164 | 1.835% |

- P3 wins versus B1: 1/2; two-task macro gain: 0.539%.
- Correct CLS FiLM control is best on both tasks: False.
- FiLM branch non-degenerate: True.
- Token representation cosine below 0.99: True.
- FiLM gradients active: True.
- Added parameters versus B1: 1.154% (<5%: True).

## Required questions

1. Global improves the token representation: No clear validation improvement.
2. The tested gain is from feature refinement rather than attention selection; GCTR keeps the B1 attention path unchanged and changes only the post-aggregation token representation.
3. CLS provides useful conditioning beyond the fixed token representation: Not reliably under the control test.
4. Worth extending to four tasks: No.
5. Potential to replace Dual ensemble: Not supported by this screen.

## Mechanism diagnostics

| Task | mean(|gamma-1|) | std(gamma) | mean(|beta|) | refinement ratio | cos(ht, refined) | L2 distance |
|---|---:|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.019784 | 0.024727 | 0.009129 | 0.028204 | 0.967696 | 3.988150 |
| MP-Bandgap | 0.011983 | 0.015427 | 0.005001 | 0.015916 | 0.856404 | 10.026794 |

## Protocol integrity

- Validation-only checkpoint selection and reporting.
- FiLM controls use the selected P3 checkpoint without retraining.
- No ensemble, averaging, router, alpha, temperature, loss-weight, or TAGIF tuning.
- `test_forward_count = 0` for every run and control.
