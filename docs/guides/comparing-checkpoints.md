# Comparing two checkpoints

The most common question: did the new checkpoint get better? Here is a study that answers
it fairly.

## 1. Set up the study

```console
$ fieldtrial init ckpt-compare
```

Edit `ckpt-compare/study.yaml`. Two arms, the same serving settings, different
checkpoints:

```yaml
arms:
  - id: step40k
    label: "40k steps"
    policy: {path: outputs/run7/checkpoints/040000/pretrained_model}
  - id: step60k
    label: "60k steps"
    policy: {path: outputs/run7/checkpoints/060000/pretrained_model}

conditions:
  factors:
    slot: {range: [1, 40]}     # 40 marked starting positions

design:
  type: randomized_block
  blinding: operator
  seed: 20261101               # any fixed number; pick a new one per study

analysis:
  primary:
    comparison: {treatment: step60k, control: step40k}
    alternative: two-sided
    alpha: 0.05
```

Choose `two-sided` unless you would genuinely act the same way on "worse" as on "no
difference". Decide it now: the alternative is part of the locked design.

## 2. Check what it can detect, then lock

```console
$ fieldtrial plan ckpt-compare --baseline 0.7
$ fieldtrial lock ckpt-compare
```

## 3. Run it

```console
$ fieldtrial serve ckpt-compare --lan
```

Each block runs both checkpoints from the same starting position, in random order. The
console tells the operator which blind code to load. Keep sessions short enough that the
rig does not drift, and use the rig checklist.

## 4. Read the result

```console
$ fieldtrial unblind ckpt-compare
$ fieldtrial report ckpt-compare
```

The primary analysis is the exact McNemar test on complete blocks, with a Tango interval
for the paired difference. The report also gives the independent-samples comparison as a
sensitivity check, the stage funnel (where each checkpoint fails), timing, and drift
checks.

If the test rejects, the report states the difference and its interval. If not, it says
"No significant difference detected" and how large a difference the study could have
detected. Neither outcome says one checkpoint is "better" in general: only that, on this
task and these conditions, one succeeded more often.
