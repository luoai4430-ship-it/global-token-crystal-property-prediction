# Original SFTMAT Audit

## Verdict

`Original SFTMAT = graph + text`

`NOT graph-only`

The formal baseline in this benchmark is the project's original
`model.net.SFTGNNMultimodal` with `property_fusion="concat"`. The graph-only
`SFTGNN` class is a separate auxiliary model and is not used as the baseline.

## Audited implementation

- Project root: `/root/autodl-tmp/TextResidualSHFMat`
- Formal model class: `model/net.py::SFTGNNMultimodal`
- Graph encoder class: `model/net.py::SFTGNN`
- Text projection class: `model/net.py::TextMLP`
- Historical command entry point: `run.py --model concat`
- Historical training/evaluation implementation: `train_and_test.py`
- Formal configuration: `model/config.py` and `data/config.py`

## Architecture

### Graph encoder

The crystal branch is the unmodified five-layer `SFTGNN` encoder:

- CGCNN atom input features: 92 dimensions.
- Atom projection: `Linear(92, 128)`.
- Five `SFTConv` message-passing layers.
- Edge features combine a 64-dimensional radial-basis distance embedding and
  a 64-dimensional projection of spherical harmonics with `l=4`.
- Mean scatter pooling produces one 128-dimensional graph representation per
  crystal.

### Text feature and text encoder

The original multimodal model consumes a frozen 768-dimensional MatBERT CLS
feature. `TextMLP` applies:

`Linear(768, 128) -> ReLU -> Linear(128, 64)`.

For the unified benchmark, the CLS feature is read from position zero of the
same frozen MatBERT `last_hidden_state [N, 128, 768]` cache used by the latest
model. This is an input adapter only; the audited `SFTGNNMultimodal` class and
all of its trainable modules remain unchanged.

### Fusion and prediction head

The graph representation (128) and projected text representation (64) are
concatenated into a 192-dimensional vector. The original property head is:

`Linear(192, 256) -> SiLU -> Linear(256, 1)`.

There is no residual addition in this baseline.

### Parameter count

The audited baseline has exactly **2,661,154 trainable parameters** under the
formal 92-dimensional CGCNN atom-feature configuration.

## Historical training protocol audit and reuse decision

The historical entry point configures the baseline through:

```text
python run.py --task train_and_test --target <TASK> --model concat \
  --text_emb_path <MatBERT_embeddings.pt>
```

The historical optimizer is AdamW with `lr=1e-3`, `max_lr=1e-3`,
`weight_decay=1e-4`, and OneCycleLR. The property loss is L1/MAE and historical
checkpoints are ranked by validation MAE.

The historical dataset loader uses `drop_last=True` for training. The user's
paper comparison is intentionally preserved rather than silently changing that
published baseline. Original SFTMAT is therefore **not retrained** in the
current run. Its frozen paper Test results are read from
`paper_tables/strict_main_results.csv` and the retained MBJ seed-level table.

Nine non-MBJ rows are the single formal paper runs. Their aggregate summaries
and raw-log paths survive, although the raw logs themselves were not cloned
into this container. MBJ is the exception: the paper main table reports a
six-seed mean, while the retained per-seed table contains an exact seed-42
Original SFTMAT result. The current seed-42 comparison uses that exact MBJ row
and preserves the six-seed paper mean as provenance metadata.

## Unified benchmark safeguards

- The runner imports and instantiates `model.net.SFTGNNMultimodal` directly.
- It does not reimplement or approximate the baseline architecture.
- Only the latest model is newly trained, once per task at seed `42`.
- The latest model uses 300 epochs, AdamW, OneCycleLR, batch size 32, BF16
  autocast, and `drop_last=False` for every split.
- Bandgap predictions are clipped to be nonnegative for validation and test.
- Checkpoints are selected only by validation MAE.
- Test is disabled until all 10 new best-validation checkpoints have been frozen.
- Each frozen checkpoint can be evaluated on Test exactly once.
- The final comparison identifies historical baseline rows explicitly and does
  not claim they were retrained under the new loader protocol.

## Latest model lock

The comparison model is the already selected `TrueTokenTextResidual` variant:

- MatBERT true token `last_hidden_state [N, 128, 768]`.
- Graph-conditioned token attention.
- L2-normalized query and keys (cosine similarity).
- Fixed temperature `tau=0.5`.
- PAD, CLS, and SEP excluded from attention.
- Graph prediction plus `residual_head([h_graph, h_text])`.
- Frozen loss: `MAE(final, target) + 0.2 * MAE(graph, target)`.

No temperature, attention, loss, optimizer, scheduler, or model-structure
search is permitted in the final benchmark.
