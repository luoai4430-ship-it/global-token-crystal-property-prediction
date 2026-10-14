# SBDVF extra-task baseline audit

Date: 2026-09-14

## Conclusion

The values `0.120612` for JARVIS-Bandgap_OPT and `0.03549224` for MP-BulkModuli are not validation MAEs in the matched Original SFTMAT records. In `strict_best_valid_server_audit.csv`, they appear as `strict_test_mae` for the corresponding Original SFTGNNMultimodal runs. The matched validation MAEs in those same records are `0.11897972` and `0.03995199`, respectively.

Therefore, the former SBDVF extra-task comparisons (`+0.419%` and `-7.267%`) are removed from the main text. No replacement SBDVF gain is reported because the local SBDVF extra-task result files do not provide a complete, independently verifiable record of the validation split, seed, checkpoint, and target-transform match needed for a fair comparison.

## Evidence

| Task | Former paper value | Evidence field | Matched validation value | Evidence source | Split / seed status |
|---|---:|---|---:|---|---|
| JARVIS-Bandgap_OPT | 0.12061202 | `strict_test_mae` | 0.11897972 | `strict_best_valid_server_audit.csv`, source log `SFTGNNMultimodal-concat-Jarvis-Bandgap_OPT.log`, checkpoint `...valid0.118980_test0.120612...pth` | JARVIS fixed split in manuscript; seed 42 is the frozen main-run protocol, but the audit CSV row itself does not contain a seed column |
| MP-BulkModuli | 0.03549224 | `strict_test_mae` | 0.03995199 | `strict_best_valid_server_audit.csv`, source log `SFTGNNMultimodal-concat-MP-BulkModuli.log`, checkpoint `...valid0.039952_test0.035492...pth` | MP fixed split and log10(GPa) target space in manuscript; seed 42 is the frozen main-run protocol, but the audit CSV row itself does not contain a seed column |

## Protocol notes

- Both records use validation-MAE checkpoint selection, followed by a separate test evaluation; the checkpoint filenames encode both values.
- MP bulk modulus uses the log10(GPa) target transformation described in `main_v10.tex`.
- The original SBDVF six-task comparison remains separate and uses the corrected matched validation reference for MP-FormationEnergy (`0.0204598904`) from `sbdvf_6task_branch_diagnosis_corrected.csv`.
- The incomplete residual SEC-SFT MP-Bandgap run is not used as evidence in the revised manuscript.
