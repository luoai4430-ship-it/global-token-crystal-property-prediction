# Dataset package

This directory contains the small, redistributable part of the dataset
provenance package. It does not contain crystal structures, graph caches,
MatBERT embeddings, model checkpoints, or target files copied from upstream
providers.

## Sources and canonical inputs

The experiments use task-specific snapshots derived from JARVIS and the
Materials Project. On the restored server, the canonical task input is
`<task>/cif/id_prop.csv`; the task description file is
`<task>/description.csv`. The corresponding server root is:

```text
/root/autodl-tmp/dataset1/dataset
```

The tracked `manifests/task_dataset_manifest.csv` records the source paths,
column names, units, row counts, and SHA-256 values for those files. The
per-task split files preserve sample IDs and `splitSeed123` assignments for
the source snapshot.

## Formal benchmark sizes

The main processed benchmark uses the sizes recorded in
`manifests/formal_split_sizes.csv`:

| Task group | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| JARVIS Bandgap MBJ | 14,537 | 1,817 | 1,818 |
| JARVIS Bandgap OPT, Formation Energy, Total Energy | 44,578 | 5,572 | 5,573 |
| JARVIS Bulk/Shear Modulus | 15,744 | 1,968 | 1,968 |
| MP Bandgap, Formation Energy | 60,000 | 5,000 | 516 |
| MP Bulk/Shear Moduli | 4,664 | 583 | 583 |

The MP Bandgap and Formation Energy source snapshot contains 65,516 rows,
whereas the formal processed benchmark has 60,000/5,000/516. The tracked
80/10/10 files for those two tasks are therefore reference manifests for the
source snapshot, not a claim that the formal processed split was regenerated
from that simple rule. Formal sample-ID checks must use the processed cache
and prediction files from the experiment output.

The split seed is `123`, which is distinct from the model-training seeds
`42` and `43`.

## Validation

With the canonical dataset mounted, run:

```bash
python data/scripts/validate_dataset_layout.py \
  --root /root/autodl-tmp/dataset1/dataset \
  --manifest data/manifests/task_dataset_manifest.csv
```

The command checks task directories, required files, unique sample IDs, and
the recorded source-manifest counts. It does not download upstream data or
run model training or test inference.
