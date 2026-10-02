# Checkpoint ladders

Several checkpoints from one training run (a "ladder") answer a different question from
two checkpoints: where does performance level off?

```console
$ fieldtrial init ladder --template checkpoint-ladder
```

The template has four checkpoints of one run on 20 starting positions, with the last
checkpoint as the pre-registered treatment and the first as the control:

```yaml
arms:
  - {id: step-010k, label: "10k steps", policy: {path: outputs/run/checkpoints/010000/pretrained_model}}
  - {id: step-020k, label: "20k steps", policy: {path: outputs/run/checkpoints/020000/pretrained_model}}
  - {id: step-030k, label: "30k steps", policy: {path: outputs/run/checkpoints/030000/pretrained_model}}
  - {id: step-040k, label: "40k steps", policy: {path: outputs/run/checkpoints/040000/pretrained_model}}

analysis:
  primary:
    comparison: {treatment: step-040k, control: step-010k}
  multiplicity: holm
  ladder:
    arms: [step-010k, step-020k, step-030k, step-040k]
    steps: [10000, 20000, 30000, 40000]
    margin: 0.10
```

With more than two arms the primary analysis is Cochran's Q across all arms on complete
blocks, followed by exact McNemar tests of every arm against the control with Holm's
adjustment. The pre-registered treatment's claim uses its adjusted p-value.

## Step association and plateau

The `ladder:` block pre-registers two more questions
([formulas](../stats/ladders.md)):

- **Does success change with training step?** Mantel's test of a linear association
  between step and success, stratified by condition, with `steps` as scores. Use the step
  count, or its logarithm if you expect diminishing returns, and decide before the data
  come in.
- **Where does it level off?** Each earlier checkpoint is tested for non-inferiority
  against the last one, with margin `margin` (10 percentage points above). Testing goes
  from the latest checkpoint backwards and stops at the first one that is not shown to be
  within the margin. The report names the earliest checkpoint from which every checkpoint
  was within the margin, or says that none was.

Without a `comparison`, the step-association test becomes the primary analysis:

```yaml
analysis:
  primary:
    alternative: greater      # success rises with step
  ladder:
    arms: [step-010k, step-020k, step-030k, step-040k]
    steps: [10000, 20000, 30000, 40000]
    margin: 0.10
```

The report shows success against step, with the plateau shaded, and a table of the
non-inferiority tests.

Things to keep in mind:

- **Every arm costs rollouts.** Four arms on 20 conditions is 80 rollouts, and each
  pairwise comparison has only 20 pairs. Fewer arms with more conditions usually answer
  the question better.
- **Choose the control in advance.** Comparing every checkpoint with the best-looking one
  after the fact is choosing a test after seeing the data.
- **"Not resolved" between neighbors is expected.** Adjacent checkpoints usually differ
  little; the useful signal is whether the late checkpoints beat the early ones.
- **A plateau needs precision.** Showing that a checkpoint is within 10 points of another
  takes many more rollouts than showing a 30-point gap. With 20 blocks, expect the
  plateau to be found only when the checkpoints are close to 100% or truly identical.
