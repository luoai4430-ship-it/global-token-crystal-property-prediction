# Final manuscript and source-code consistency review

Review date: 2026-10-13. Manuscript: `paper/main_v24.pdf` (17 pages),
checked against its LaTeX source. Server:
`connect.westb.seetacloud.com:36397`, canonical experiment source root
`/root/autodl-tmp/TextResidualSHFMat`.

## Verdict

The main model definitions and the explicit benchmark construction agree
with the paper's architecture and losses. The original repository was not a
self-contained reproduction release: the data package was missing, several
symlinks pointed outside the repository, and the README described a
Token-only seed42 runner as a complete paired runner. These release defects
have been corrected without changing the archived model definitions.

The cache dtype remains a manuscript discrepancy: Sections 3.5 and 4.2
describe cached MatBERT states as BF16. The server cache writer uses
`np.float16` and FP16 autocast, and `TokenFeatureStore` requires float16
arrays. BF16 applies to property-model training/evaluation autocast. The
paper was not edited during this review.

## Paper-to-code mapping

| Paper item | Source | Finding |
| --- | --- | --- |
| Independent five-layer SFTGNN encoders | `model/net.py`, runner `build_model` | Both predictors instantiate their own backbone; 92 -> 128 atom embedding, five SFTConv layers, mean pooling |
| Reciprocal-distance radial features | `model/net.py` | RBF on 1/d, 64 bins, projection to 64; same as Section 3.3 |
| Spherical-harmonic projection | `model/net.py`, `data/graph.py`, `data/utils.py` | 16 input components for l=4 -> 32 -> 64; graph cache multiplies harmonic basis by sin(theta) |
| Eq. (2) gate/message construction | `model/sublayer.py::SFTConv.message` | Gate input 320, message input 384; both hidden 512; edge-wise product before neighbor sum |
| Eq. (3) node update | `model/sublayer.py::SFTConv.forward` | Sum aggregation -> BN -> add node state -> SiLU |
| Global CLS projection | `model/net.py::TextMLP` | 768 -> 128 -> 64, ReLU; same as Section 3.4 |
| Global prediction head | `model/net.py::SFTGNNMultimodal` | Concatenate 128 graph + 64 text -> 256 -> 1, SiLU |
| CLS source | runner `OriginalSFTMATExact.forward` | Position zero of the same token sequence cache |
| Eqs. (5)-(7) normalized token attention | `model/true_token_text_residual.py` | Bias-free projections to 128, L2-normalized q/k, masked softmax, weighted value sum, LayerNorm |
| Final attention configuration | runner `build_model` | Explicit control C1 and temperature 0.5; generic class defaults do not define this configuration |
| Special-token masking | runner `special_token_ids`, model `_attention_mask` | Reads PAD/CLS/SEP IDs from the actual tokenizer and excludes them and padding |
| Eq. (8) Token prediction | model `forward` | graph_head(hg) + residual_head([hg; ht]); hidden 128/256 respectively |
| Prediction-level mean | paired summarizer | Exactly 0.5*(Global+Token), after ID and target checks |
| Training objectives | runner `train_run` | Global MAE; Token MAE(final) + 0.2*MAE(graph) |
| Optimizer/scheduler | runner `train_run` | AdamW lr1e-3, weight decay1e-4, OneCycleLR max_lr1e-3, pct_start0.3 |
| Checkpoint choice | runner `train_run`, `freeze_valid` | Minimum validation MAE; frozen before test |
| Target handling | `data/dataset.py`, runner `physical_outputs` | Train mean/std standardization; MP moduli log10(GPa); loss/MAE after inverse standardization; nonnegative bandgap clipping |
| Parameter counts | runner `build_model` guards | Global 2,661,154; Token 2,800,739; includes an unused backbone fc registered in both models |

## Server/source equality

Before repairs, all 11 existing main-method Python files matched the server
byte-for-byte, including all three model-definition files and both archived
benchmark runners. After copying the missing data modules and cache writer,
18 archived Python files match the downloaded server snapshot byte-for-byte.
SHA-256 values are in `metadata/paper_v24_server_code_comparison.json`.
This establishes source equality, not checkpoint equality or a fresh
numerical replay.

New files `run_paper_benchmark.py`, `prepare_paper_inputs.py`, and
`summarize_paper_predictions.py` are release adapters, not historical server
training files. The paired wrapper changes the allowed training seed and
dataset location while retaining the server model builder, loss, optimizer,
scheduler, freeze safeguards, and attention implementation.

## Reproduction corrections

1. Restored server `data/config.py`, `dataset.py`, `graph.py`, `utils.py`,
   `token_sequence.py`, package initializer, and `elements.json`.
2. Included the exact server token-cache writer. Its output is a directory
   containing NumPy arrays and metadata, rather than a single `.pt` file.
3. Removed broken code aliases and software links that pointed outside the
   repository. Restored actual environment files, LICENSE, and NOTICE.
4. Replaced the old README sequence with explicit fresh paired replication:
   40 runs total; 20 validation selections per seed; freeze; test once per
   selected checkpoint; saved-file summaries.
5. Retained the archived seed42 Token-only and seed43 paired runners. Their
   original sources remain unchanged. The old `run.py`/`train_and_test.py`
   import historical experimental models not included in the main release;
   README identifies them as unsupported main-paper entry points.
6. Clarified the CrysMMNet-derived dataset source, splitSeed123 versus model
   seeds42/43, MP 60000/5000/516 split, and source-reference manifest scope.

The previous release's raw-data reference manifests apply an 80/10/10 rule
to the MP source snapshot. The actual loader uses 60000/5000/remainder for
large MP tasks. Those reference manifests must not overwrite formal split
IDs. The loader provides direct evidence for this difference; this review
does not establish all earlier raw-to-eligible filtering steps.

## Checks performed and limits

- Read the final PDF methods and its corresponding source; compared model
  definitions, loss, masking, aggregation and prediction combination.
- Compared source SHA-256 values against a read-only server download.
- Python syntax compilation passed for released main-method files.
- Six release tests passed: pairing reordered IDs; rejecting duplicate
  IDs, changed targets and incomplete validation freeze. Synthetic records
  are temporary test fixtures, not reported material-property results.
  Additional wrapper checks cover both seeds, dataset-path adaptation, and
  rejection of the historical frozen-baseline summarizer.
- No training, property inference, or test-set evaluation was performed.
- Local Python lacks PyTorch/PyG. The restored server's default Python lacks
  torch_geometric and the server has no GPU. Runtime model-count/forward
  checks could not be completed; the parameter counts above are statically
  verified construction guards, not freshly executed model counts.
- Raw datasets, pretrained MatBERT weights, graph/token caches and trained
  checkpoints are not in the Git release. Exact numerical replay requires
  the historical artifacts; README documents a fresh protocol replication.

## Live GitHub comparison

The fetched GitHub main commit before this repair is
`dc0c42d8a9a7d7b5e6745d6d2efa7b38502ebe56`. The original 11 main-method
Python files in that commit match the server snapshot byte-for-byte.
The seven added Python files (the data package and cache writer) were absent
from that commit. Per-file results are included in the source-comparison
JSON. The online README still described the historical Token-only runner;
its CrysMMNet data attribution is preserved in the rewritten guide.

This review changes release completeness and reproduction instructions,
not the existing model architecture, historical predictions, or manuscript.
