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

## Historical training protocol audit

The historical entry point configures the baseline through:

```text
python run.py --task train_and_test --target <TASK> --model concat \
  --text_emb_path <MatBERT_embeddings.pt>
```

The historical optimizer is AdamW with `lr=1e-3`, `max_lr=1e-3`,
`weight_decay=1e-4`, and OneCycleLR. The property loss is L1/MAE and historical
checkpoints are ranked by validation MAE.

The historical dataset loader uses `drop_last=True` for training. This does
not satisfy the final benchmark requirement that every split use
`drop_last=False`. Historical task rows also do not provide five matched seeds
under one uniform 300-epoch protocol. Therefore no historical result is reused
for the final main comparison; the original implementation is retrained.

## Unified benchmark safeguards

- The runner imports and instantiates `model.net.SFTGNNMultimodal` directly.
- It does not reimplement or approximate the baseline architecture.
- Baseline and latest model consume identical graph splits and the same frozen
  MatBERT cache.
- Both use seeds `42, 7, 1234, 2025, 2026`, 300 epochs, AdamW, OneCycleLR,
  batch size 32, BF16 autocast, and `drop_last=False` for every split.
- Bandgap predictions are clipped to be nonnegative for validation and test.
- Checkpoints are selected only by validation MAE.
- Test is disabled until all 100 best-validation checkpoints have been frozen.
- Each frozen checkpoint can be evaluated on Test exactly once.

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
