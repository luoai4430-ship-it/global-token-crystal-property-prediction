# Global and Token-Level Crystal Property Prediction

Code accompanying the final manuscript
[main_v24.pdf](paper/main_v24.pdf):
**Global and Token-Level Crystallographic Text for Crystal Property
Prediction: Complementarity and Prediction Averaging**.

## Model implemented here

The main experiment trains two independent predictors. Each has its own
five-layer SFTGNN with 92-dimensional CGCNN atom features, 128-dimensional
node states, 64 radial features, 64 spherical-harmonic features, and mean
graph pooling. Parameters are not shared between predictors.

| Component | Implementation |
| --- | --- |
| SFTConv | Per-edge sigmoid gate times message MLP, then neighbor sum, batch normalization, residual addition, SiLU |
| Global | MatBERT CLS: 768 -> 128 -> 64 with ReLU; concatenate graph 128 + CLS 64; head 192 -> 256 -> 1 with SiLU |
| Token | Bias-free query/key/value projections to 128; L2-normalized query and keys; masked cosine softmax at tau=0.5; weighted values followed by LayerNorm |
| Token prediction | Graph head 128 -> 128 -> 1 plus residual head 256 -> 256 -> 1 |
| Final prediction | Exactly 0.5 * Global prediction + 0.5 * Token prediction, aligned by sample ID |
| Global objective | MAE(final, target) |
| Token objective | MAE(final, target) + 0.2 * MAE(graph, target) |

The model definitions are
[net.py](code/main_method/model/net.py),
[sublayer.py](code/main_method/model/sublayer.py), and
[true_token_text_residual.py](code/main_method/model/true_token_text_residual.py).
The benchmark explicitly chooses control `C1` and temperature `0.5`;
the generic class defaults are not the final paper configuration.

The MatBERT sequence cache has shape [N, 128, 768]. On disk it uses
**float16**, with input IDs and attention masks stored separately. PAD,
CLS, and SEP are excluded from Token attention using IDs read from the
checkpoint tokenizer. BF16 autocast is used during property-model training.
The manuscript's description of BF16 cache storage differs from this
implementation; see [the code review](docs/PAPER_V24_CODE_AUDIT.md).

## Repository contents

| Directory | Contents |
| --- | --- |
| `paper/` | Final manuscript, bibliography, and figures |
| `code/main_method/model/` | Paper graph, Global, and Token model definitions |
| `code/main_method/data/` | Server data loading, graph construction, split rules, and token-cache reader |
| `code/main_method/scripts/` | Cache preparation, paired training, selection freeze, and saved-prediction summaries |
| `data/manifests/` | Source-snapshot references and formal processed split sizes |
| `results/` | Published main results and validation diagnostics |
| `metadata/` | Registries and server/local source comparison |
| `software/` | Environment specifications and recorded package versions |
| `docs/` | Model/data cards, reproducibility notes, and code review |

Large upstream datasets, MatBERT weights, graph caches, token caches, and
trained checkpoints are not included. Reproduction requires the matching
data snapshot and local MatBERT checkpoint. The stored result tables can be
read without retraining. Recreating an exact historical prediction file
also requires its original cache and checkpoint.

## 1. Install the environment

The reference software uses Python 3.10, PyTorch 2.2.2, CUDA 12.1,
torch-geometric 2.5.2, torch-scatter 2.1.2, SciPy 1.15.2, and
Transformers 4.38.2. Use a CUDA GPU with BF16 support for the formal runner.

```bash
git clone https://github.com/luoai4430-ship-it/global-token-crystal-property-prediction.git
cd global-token-crystal-property-prediction
conda env create -f software/environment.yml
conda activate global-token-crystal

python -c "import torch, torch_geometric, torch_scatter, transformers; print(torch.__version__, torch.cuda.is_available(), torch.cuda.is_bf16_supported())"
```

The conda file installs the PyTorch and PyG stack together. The accompanying
`requirements.txt` lists dependencies but is not a replacement for
installing CUDA-compatible PyG extension wheels. Saved PyG caches must be
loaded in a compatible trusted environment; the reference uses PyTorch
2.2.2 rather than a newer release with different `torch.load` defaults.

## 2. Obtain the CrysMMNet-derived data snapshot

The task-specific data used by this work come from the benchmark package
used by [CrysMMNet (Das et al., UAI 2023)](https://proceedings.mlr.press/v216/das23a.html).
The underlying structures and targets originate from JARVIS and Materials
Project. Preserve the snapshot's structures, targets, descriptions, and IDs;
independent current downloads of those databases need not produce the same
samples.

Set `DATA_ROOT` to the snapshot directory and `MATBERT` to the local
MatBERT checkpoint, containing model weights, configuration, and tokenizer.

```bash
export DATA_ROOT=/path/to/crysmmnet-derived/dataset
export MATBERT=/path/to/matbert-base-cased
export WORK=/absolute/path/to/paper_replication
mkdir -p "$WORK"
```

Each task directory must contain `cif/id_prop.csv` (headerless ID and raw
target), the corresponding CIF files, and `description.csv`.

| Task | Directory under DATA_ROOT | Description ID/text columns |
| --- | --- | --- |
| Jarvis-Bandgap_MBJ | jarvis/mbj_bandgap | Id / Description |
| Jarvis-Bandgap_OPT | jarvis/optb88vdw_bandgap | Id / Description |
| Jarvis-FormationEnergy | jarvis/formation_energy_peratom | Id / Description |
| Jarvis-TotalEnergy | jarvis/optb88vdw_total_energy | Id / Description |
| Jarvis-BulkModulusKv | jarvis/bulk_modulus_kv | Id / Description |
| Jarvis-ShearModulusGv | jarvis/shear_modulus_gv | Id / Description |
| MP-Bandgap | mp_2018/MP-bandgap | id / text |
| MP-FormationEnergy | mp_2018/MP-formation | id / text |
| MP-BulkModuli | mp_2018_small/bulk | Id / Description |
| MP-ShearModuli | mp_2018_small/shear | Id / Description |

The restored server stores the snapshot under
`/root/autodl-tmp/dataset1/dataset`. Adapt paths for your machine.

## 3. Prepare graph splits and the token cache

Run all model commands from `code/main_method`. The data loader uses fixed
**split seed 123**, distinct from training seeds 42/43. JARVIS uses
80%/10%/remainder. Large MP tasks use 60,000/5,000/remainder; the smaller
elastic datasets use 80%/10%/remainder. MP elastic targets are transformed to
log10(GPa); JARVIS elastic targets remain in GPa. Targets are standardized
using the training mean and standard deviation, then predictions are mapped
back to the benchmark target space for loss and evaluation.

```bash
cd code/main_method
mkdir -p Dataset/raw Dataset/processed

TASKS=(
  Jarvis-Bandgap_MBJ Jarvis-Bandgap_OPT Jarvis-FormationEnergy
  Jarvis-TotalEnergy Jarvis-BulkModulusKv Jarvis-ShearModulusGv
  MP-Bandgap MP-FormationEnergy MP-BulkModuli MP-ShearModuli
)

for task in "${TASKS[@]}"; do
  python scripts/prepare_paper_inputs.py \
    --task "$task" --data_root "$DATA_ROOT" \
    --output_csv "$WORK/inputs/$task.csv"

  python scripts/cache_matbert_token_features.py \
    --input "$WORK/inputs/$task.csv" \
    --id_column sample_id --text_column text \
    --model_name "$MATBERT" --max_length 128 --batch_size 32 \
    --output_dir "$WORK/token_cache/$task" --device cuda
done
```

The first command builds or loads graph caches and exports descriptions in
processed split-ID order. Missing descriptions cause an error; obtain the
matching snapshot rather than dropping samples. The second command writes a
cache directory, **not a .pt file**, containing `token_features.npy`,
`attention_mask.npy`, `input_ids.npy`, `sample_ids.json`, and
`metadata.json`. Global reads token position zero from this same cache.

Check the emitted `*.split.json` counts against
`data/manifests/formal_split_sizes.csv` (relative to the repository root).
In particular, MP-Bandgap and MP-FormationEnergy have 60,000/5,000/516
samples in the paper. The MP 80/10/10 source-reference files in
`data/manifests/splits/` are not the formal benchmark partition.
Do not regenerate paper results from those reference assignments.

## 4. Train both predictors for each seed

A fresh paired replication has **40 training runs**:
10 tasks x 2 predictors x 2 training seeds. Keep a separate output root for
each seed. The wrapper below delegates to the unchanged server paired
runner, adapting only the allowed seed and dataset path.

```bash
for seed in 42 43; do
  ROOT="$WORK/seed$seed"
  for task in "${TASKS[@]}"; do
    python scripts/run_paper_benchmark.py \
      --training_seed "$seed" --data_root "$DATA_ROOT" preflight \
      --task "$task" --token_cache "$WORK/token_cache/$task" \
      --scratch "$WORK/scratch/$task" --output_root "$ROOT" \
      --model_checkpoint "$MATBERT"

    for model in original_sftmat ours; do
      python scripts/run_paper_benchmark.py \
        --training_seed "$seed" --data_root "$DATA_ROOT" train \
        --task "$task" --model_name "$model" --seed "$seed" \
        --token_cache "$WORK/token_cache/$task" \
        --output_dir "$ROOT/runs/$task/$model/seed_$seed" \
        --model_checkpoint "$MATBERT" --device cuda \
        --num_workers 2 --epochs 300 --batch_size 32
    done
  done
done
```

`original_sftmat` names the Global model; `ours` names the Token model.
The fresh paired runner uses AdamW, lr/max_lr 1e-3, weight decay 1e-4,
OneCycleLR (`pct_start=0.3`), BF16 autocast, two loader workers,
`drop_last=False`, training batch 32, and evaluation batch 64. It selects
the minimum validation-MAE checkpoint and refuses to overwrite completed
runs. Bandgap predictions are clipped to nonnegative values.

Preflight evaluates a few validation samples and checks parameter counts,
attention masks, cache coverage, and numerical finiteness. It does not run
test inference. Per seed, expect 20 validation summaries and checkpoints,
with `test_forward_count=0` before selection freeze.

The historical `run_final_10task_benchmark.py` trains Token for seed 42
against frozen Global references. It does not train both main predictors.
`run.py` and `train_and_test.py` are historical entry points with imports
from excluded experimental models; use the paired wrapper above for the
paper architecture. Historical seed42 references may have their original
loader settings; a fresh paired replication does not retroactively change
those published results.

## 5. Freeze validation selection and evaluate test once

After both seed roots contain all 20 completed validation runs:

```bash
for seed in 42 43; do
  python scripts/run_paper_benchmark.py --training_seed "$seed" freeze \
    --output_root "$WORK/seed$seed"
done
```

Each freeze manifest must contain exactly the ten tasks and two predictors
for its seed. The runner verifies protocol and checkpoint hashes. Keep
output paths stable after freezing because the manifest records checkpoint
paths.

Then evaluate each frozen checkpoint once:

```bash
for seed in 42 43; do
  for task in "${TASKS[@]}"; do
    for model in original_sftmat ours; do
      python scripts/run_paper_benchmark.py \
        --training_seed "$seed" --data_root "$DATA_ROOT" test \
        --task "$task" --model_name "$model" --seed "$seed" \
        --token_cache "$WORK/token_cache/$task" \
        --output_root "$WORK/seed$seed" \
        --model_checkpoint "$MATBERT" --device cuda --num_workers 2
    done
  done
done
```

The runner refuses test evaluation before freeze or after test has already
started for that checkpoint. The counter denotes one full test evaluation,
not one mini-batch. Test labels are excluded from checkpoint selection and
weight fitting.

## 6. Recompute the prediction mean from saved files

```bash
python scripts/summarize_paper_predictions.py \
  --run "42=$WORK/seed42" --run "43=$WORK/seed43" \
  --split test --output_dir "$WORK/summary_test"

python scripts/summarize_paper_predictions.py \
  --run "42=$WORK/seed42" --run "43=$WORK/seed43" \
  --split valid --output_dir "$WORK/summary_valid"
```

These commands load saved CSVs only. They align predictions by ID, reject
duplicate IDs, different sample sets or targets, calculate the fixed
0.5/0.5 mean, and emit paired CSVs, per-seed task results, two-run mean/sample
standard deviation, macro gains, and win counts. Macro gains average
task-wise relative MAE reductions; raw MAEs with different units are not
averaged. The manuscript reports about 5.9% macro gain over Global, 10/10
wins over Global and 9/10 wins over both constituents per run. Exact replay
needs the historical checkpoint/cache snapshots as well as these sources.

## Validation-only studies

The matched CLS/mean/cosine ablation and interaction studies have their own
task coverage and budgets. Their results are in `results/diagnostics/`;
they are not additional two-seed test models. The main code is intended for
the independent Global and Token predictors. The release does not claim to
include a runnable implementation for every exploratory appendix variant.

## Source consistency and remaining limitations

The model files and archived benchmark runners were compared with
`/root/autodl-tmp/TextResidualSHFMat` on
`connect.westb.seetacloud.com:36397`. Results and file hashes are recorded
in `metadata/paper_v24_server_code_comparison.json` and
`docs/PAPER_V24_CODE_AUDIT.md`. New reproduction wrappers are separate
from the unchanged historical server code. The restored container currently
lacks `torch_geometric` and has no GPU, so this review did not perform
training or test inference.

## Authors and citation

Huaijuan Zang; Yunfan Peng (corresponding author, luoai4430@gmail.com);
Chong Zhao; Fan Yang; Feng Hong; Liangfeng Xu.
See the manuscript for affiliations and funding, and `CITATION.cff` for
repository citation. Dataset attribution follows CrysMMNet:

> Kishalay Das, Pawan Goyal, Seung-Cheol Lee, Satadeep Bhattacharjee, and
> Niloy Ganguly. CrysMMNet: Multimodal Representation for Crystal Property
> Prediction. Proceedings of UAI, PMLR 216:507-517, 2023.

OpenAI Codex assisted with code preparation and organization. The authors
reviewed the code and remain responsible for the research and results.
See `LICENSE` and `NOTICE`; upstream datasets and pretrained weights
retain their own terms.
