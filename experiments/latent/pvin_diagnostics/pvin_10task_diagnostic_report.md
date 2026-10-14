GT_DOMINANT_SIGNAL

# PVIN 10-task interaction comprehensive diagnostic

- zero-all positive count: 10/10
- branch positive counts: GC 7/10, GT 10/10, CT 9/10
- mean contributions: GC 0.536%, GT 1.375%, CT 0.441%, ALL 3.494%
- sample helpful fraction mean: GC 0.278, GT 0.292, CT 0.253, ALL 0.325

## Interpretation

u_gt remains the most stable branch. PVIN failures are not explained by one universally harmful branch or by raw magnitude alone; interactions are useful inside the trained head, but they do not reliably improve over AuxConcat on every property.

The only structurally justified next check is a uniform minimal-interaction variant that keeps graph-token interaction and removes the less stable graph-CLS and CLS-token branches. This is not property-specific routing.

TRAINING_RUN_COUNT_NEW = 0
TEST_FORWARD_COUNT = 0
