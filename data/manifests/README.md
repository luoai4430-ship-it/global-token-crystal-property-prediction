These manifests were generated from the restored server snapshot on
2026-09-25. They are ID-keyed source-snapshot references, not copies of the
crystal structures or text caches. The canonical source file for each task is
`cif/id_prop.csv`; `description.csv` is recorded for text provenance.

The reference split files use `random.seed(123)` with 80/10/10 truncation.
For MP-Bandgap and MP-FormationEnergy, this reconstructed source-snapshot
partition is **not** the formal processed benchmark partition used by the
paper: the formal processed sizes are 60000/5000/516. Use
`formal_split_sizes.csv` for the benchmark sizes and do not replace the
processed-cache IDs with the reconstructed reference split.
