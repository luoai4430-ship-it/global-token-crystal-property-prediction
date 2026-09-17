# Shared-Backbone Dual-View Fusion 6-task validation experiment

Decision: **WEAK_SBDVF_SIGNAL**

Validation-only seed42 experiment. No test evaluation was run.

## Results

| Task | Original | Shared Global | Shared Token | SBDVF final | Gain vs Original | Final vs best branch |
|---|---:|---:|---:|---:|---:|---:|
| Jarvis-Bandgap_MBJ | 0.240073875 | 0.247643807 | 0.244045529 | 0.241590648 | -0.632% | 1.006% |
| Jarvis-FormationEnergy | 0.0289658173 | 0.028823118 | 0.0286445492 | 0.028504994 | 1.591% | 0.487% |
| Jarvis-BulkModulusKv | 9.21483803 | 8.93980408 | 8.88783345 | 8.86950076 | 3.748% | 0.206% |
| MP-Bandgap | 0.212995555 | 0.204922247 | 0.202905401 | 0.200831884 | 5.711% | 1.022% |
| MP-FormationEnergy | 0.02388391 | 0.0201996003 | 0.0200497132 | 0.0199049795 | 16.659% | 0.722% |
| MP-ShearModuli | 0.0658285211 | 0.066835065 | 0.0685223846 | 0.0669763828 | -1.744% | -0.211% |

## Required answers

1. Shared Backbone Dual-View beats Original on 4/6 tasks.
2. Six-task macro gain vs Original: 4.222%.
3. JARVIS macro: 1.569%; MP macro: 6.876%.
4. Electronic/Energetic/Mechanical macro: 2.539% / 9.125% / 1.002%.
5. Final beats the best shared single branch on 5/6 tasks.
6. Mean oracle gain vs best shared branch: 10.190%.
7. Mean gradient cosine: 0.325; mean negative fraction: 0.267.
8. VIEW_COLLAPSE: False.
9. SBDVF/IndependentDual parameter ratio: 0.541.
10. test_forward_count = 0.
