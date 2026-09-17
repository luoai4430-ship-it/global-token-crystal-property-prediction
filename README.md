# Global--Token crystal property prediction

This repository accompanies the `main_v21` English and Chinese manuscripts.
The primary study compares two independently trained graph--text prediction
pathways and combines their **sample-aligned predictions** with a fixed
0.5/0.5 mean:

1. **Global pathway:** SFTGNN graph encoding with projected MatBERT CLS text
   features and a scalar regression head.
2. **Token pathway:** the same SFTGNN backbone with graph-conditioned,
   cosine-normalized attention over frozen MatBERT contextual token features,
   followed by the token-level predictor described in the paper.
3. **Global+Token:** the fixed prediction mean after each pathway has selected
   its checkpoint using validation MAE only.

The main protocol uses ten tasks, two independent training runs (seeds 42 and
43), and one final test evaluation for each validation-selected checkpoint.
The matched text-representation and interaction-location studies are
validation-only diagnostics; they are not additional main test models.

## Authors

- Huaijuan Zang, School of Computer and Information, Hefei University of Technology, zanghj@hfut.edu.cn
- Yunfan Peng, School of Computer and Information, Hefei University of Technology (corresponding author)
- Chong Zhao, Engineering Quality Education Center, Hefei University of Technology, zhaochong@hfut.edu.cn
- Fan Yang, School of Computer and Information, Hefei University of Technology, 2021800201@hfut.edu.cn
- Feng Hong, Chizhou University, hongfeng@czu.edu.cn
- Liangfeng Xu, School of Computer and Information, Hefei University of Technology, xulfcjn@hfut.edu.cn

## Funding

This work was supported by the Industry--University Cooperation Collaborative
Education Project of the Ministry of Education (No. 250603873093026), the
Fundamental Research Funds for the Central Universities of China (Grant No.
PA2025GDSK0036), and the Innovative Teaching Team for Ideological and
Political Education in Electronic Information Science and Technology
(No. 2025XKSTD01).

## Repository layout

| Path | Contents |
| --- | --- |
| `paper/` | English and Chinese manuscripts, bibliography, and figures |
| `code/main_method/` | Current data loaders, models, benchmark runners, and summarizers |
| `results/main_test/` | Main ten-task test tables and selection metadata |
| `results/seed42/` | Seed-42 paired results |
| `results/seed43/` | Seed-43 paired results and two-run summaries |
| `results/diagnostics/` | Declared validation-only diagnostic reports |
| `metadata/` | Dataset/task registry, split information, schemas, and checksums |
| `docs/` | Data card, model card, reproducibility notes, and provenance notes |
| `software/` | Environment and software records |

Raw datasets, graph caches, MatBERT caches, and model checkpoints are not
duplicated in this repository. They must be obtained under the terms of the
upstream dataset providers and placed at paths configured for the local
machine. The release intentionally does not publish the earlier
`TextResidualSHFMat` residual-fusion model as the current method.

## Main protocol

The following values are locked for the main benchmark:

| Setting | Value |
| --- | --- |
| Tasks | 6 JARVIS properties and 4 Materials Project properties; see `metadata/dataset_registry.csv` |
| Data partition | fixed `splitSeed123` |
| Training seeds | `42` and `43` |
| Epochs | `300` |
| Batch size | `32` |
| Precision | bfloat16 |
| Optimizer | AdamW, learning rate `1e-3`, weight decay `1e-4` |
| Scheduler | OneCycleLR, maximum learning rate `1e-3` |
| Token sequence length | `128` |
| Token attention temperature | `0.5` |
| Checkpoint selection | minimum validation MAE |
| Test evaluation | once per validation-selected checkpoint, after validation freeze |

The partition seed is a data-splitting parameter. It is distinct from the
model-training seeds. Do not substitute seed 42 or 43 for `splitSeed123`.

## Reproduce the main benchmark

### 1. Create the environment

Use Python 3.10 or a compatible PyTorch/torch-geometric environment. The
recorded environments are in `software/environment.yml`,
`software/requirements.txt`, and `software/pip-freeze.txt`.

```bash
git clone https://github.com/luoai4430-ship-it/global-token-crystal-property-prediction.git
cd global-token-crystal-property-prediction

# Example conda setup; adapt the CUDA channel to the installed driver.
conda env create -f software/environment.yml
conda activate <environment-name>
```

Verify the installation before a long run:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 2. Prepare data and caches

Prepare the upstream JARVIS and Materials Project structures, task targets,
fixed `splitSeed123` partitions, graph caches, and the length-128 MatBERT
token cache. Keep the sample-ID columns intact. The task locations used by the
benchmark scripts are declared near `TASKS` in
`code/main_method/scripts/run_final_10task_benchmark.py`; update those local
paths for your machine rather than changing the task definitions or split.

The token cache must contain the union of the train, validation, and test IDs
for the task. Before training, verify that graph-cache IDs and token-cache IDs
match exactly. The corresponding cache metadata and checksums should be saved
with the run output.

### 3. Run one validation training

The formal benchmark runner is:

```bash
cd code/main_method
python scripts/run_final_10task_benchmark.py train \
  --task Jarvis-Bandgap_MBJ \
  --model_name ours \
  --seed 42 \
  --token_cache /path/to/matbert_token_cache.pt \
  --output_dir /path/to/output/runs/Jarvis-Bandgap_MBJ/ours/seed_42 \
  --model_checkpoint /path/to/matbert-checkpoint \
  --device cuda \
  --num_workers 0 \
  --epochs 300 \
  --batch_size 32
```

Repeat this command for every task and for both training seeds. The runner
records validation predictions, the selected checkpoint, protocol metadata,
and `summary.json`. It refuses to overwrite a completed run.

The repository names the current token model `ours` in the benchmark code.
This is the TrueToken pathway in the manuscript. The independent Global
baseline and its frozen reference results are recorded in the result and
provenance files; they must be evaluated with the same split, seed, and
checkpoint-selection rule before forming the fixed prediction mean.

### 4. Freeze validation selection

After all 20 model/seed/task validation runs finish, freeze the validation
selection before any test forward pass:

```bash
python scripts/run_final_10task_benchmark.py freeze \
  --output_root /path/to/output
```

Check that the freeze manifest contains exactly 20 entries, all selected by
validation MAE, and that every pre-test `test_forward_count` is zero.

### 5. Evaluate the selected checkpoints once on test

Only after the freeze, run one test evaluation per selected checkpoint:

```bash
python scripts/run_final_10task_benchmark.py test \
  --task Jarvis-Bandgap_MBJ \
  --model_name ours \
  --seed 42 \
  --token_cache /path/to/matbert_token_cache.pt \
  --output_root /path/to/output \
  --model_checkpoint /path/to/matbert-checkpoint \
  --device cuda \
  --num_workers 0
```

Repeat for all frozen entries. Do not use test labels for model selection,
early stopping, hyperparameter tuning, or deciding which checkpoint to keep.

### 6. Summarize the paired predictions

After the 20 test summaries are present, summarize the fixed Global+Token
prediction mean:

```bash
python scripts/run_final_10task_benchmark.py summarize \
  --output_root /path/to/output \
  --baseline_csv /path/to/matched/original_baseline.csv
```

The output should include the per-task table, per-seed summaries, paired
sample-aligned predictions, macro statistics, and the final selection record.
Use `results/main_test/` and `results/seed43/` as format references. The
standalone helper
`code/main_method/scripts/summarize_seed43_dualview.py` checks paired seed-43
IDs and computes the Global, Token, and fixed-mean summaries for an already
completed seed-43 output root.

## Validation-only diagnostic studies

The matched CLS/masked-mean/cosine comparison and the message-, latent-, and
shared-backbone interaction studies are exploratory validation experiments.
Their coverage and seeds are declared in `metadata/experiment_registry.csv`.
Do not merge their metrics into the main ten-task test table, and do not call
them two-run test evidence unless the corresponding protocol was actually
repeated.

## Reproducibility checks

Before reporting a run, confirm:

1. `splitSeed123` is unchanged and is distinct from the training seed.
2. Graph, target, and token-cache sample IDs match within every split.
3. The selected checkpoint has the minimum validation MAE.
4. No test forward occurred before validation freeze.
5. Global and Token predictions are aligned by sample ID before averaging.
6. The fixed fusion weight is exactly `0.5/0.5`; no router, gate, or
   sample-wise weight is introduced.
7. The reported protocol and file hashes agree with `metadata/`.

## Citation and license

See `paper/cas-refs.bib`, `software/CITATION.cff`, `LICENSE`, and `NOTICE`.
Raw data remain subject to the licenses and access conditions of JARVIS and
Materials Project.
