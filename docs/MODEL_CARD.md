# Model card: Global and Token graph--text predictors

## Intended use

The released models estimate scalar crystal properties from a periodic crystal
graph and a Robocrystallographer-derived crystallographic description. They are
research models for benchmark comparison and rapid screening, not certified
tools for safety-critical or experimental decisions.

## Primary architecture

The paper's main study contains two independently initialized and trained
predictors with the same SFTGNN graph encoder:

- **Global**: SFTGNN graph representation concatenated with a projected
  MatBERT CLS representation.
- **Token**: graph-conditioned attention over frozen MatBERT contextual token
  states. Query and key vectors are L2-normalized and the temperature is fixed
  at `tau=0.5`; padding and special tokens are excluded.

Each predictor selects its checkpoint by validation MAE. The reported
Global+Token prediction is the sample-aligned fixed mean
`0.5 * Global + 0.5 * Token`. This is prediction-level combination, not a
single latent fusion block, router, or learned mixture weight.

## Training and evaluation

- data partition: fixed `splitSeed123`;
- model-training seeds: `42` and `43`;
- maximum training budget: 300 epochs;
- optimizer: AdamW with OneCycleLR and maximum learning rate `1e-3`;
- weight decay: `1e-4`;
- frozen MatBERT cache: maximum sequence length 128;
- checkpoint criterion: minimum validation MAE;
- main evaluation: one test forward pass per validation-selected predictor.

## Scope of auxiliary studies

The matched CLS/masked-mean/cosine representation ablation and the
message-level, latent-level, and shared-backbone interaction experiments are
validation-only diagnostic studies. They should not be interpreted as
additional final models or as a complete ranking of fusion locations.

## Historical code separation

Files named `text_residual_shfmat.py` and related residual-fusion experiments
belong to the earlier TextResidualSHFMat line and are excluded from this
release. They are not the current paper's main implementation.

## Limitations

The results cover the stated datasets, targets, splits, and two independent
training seeds. Generated descriptions may contain omissions or systematic
vocabulary biases. The fixed prediction average requires evaluating both
pathways and therefore has higher inference cost than either constituent
predictor alone.
