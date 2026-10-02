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
```

With more than two arms the primary analysis is Cochran's Q across all arms on complete
blocks, followed by exact McNemar tests of every arm against the control with Holm's
adjustment. The pre-registered treatment's claim uses its adjusted p-value.

Things to keep in mind:

- **Every arm costs rollouts.** Four arms on 20 conditions is 80 rollouts, and each
  pairwise comparison has only 20 pairs. Fewer arms with more conditions usually answer
  the question better.
- **Choose the control in advance.** Comparing every checkpoint with the best-looking one
  after the fact is choosing a test after seeing the data.
- **"Not resolved" between neighbors is expected.** Adjacent checkpoints usually differ
  little; the useful signal is whether the late checkpoints beat the early ones.

A trend test across the ladder and plateau detection are planned for v0.2.
