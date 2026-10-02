# Reward models: pre-labels, blind review and proxy estimates

A reward model watches an episode and estimates whether the task was done. fieldtrial uses
such models in three ways, and never in place of a person's label:

1. **Agreement.** Score the episodes of your trials and see how often the model agrees
   with the operator's labels (Cohen's κ). This tells you whether the model is worth using.
2. **Pre-labels with blind review.** Score extra episodes (overnight runs, DAgger
   rollouts) and let a reviewer label a random sample in the console. The reviewer labels
   first and sees the model's suggestion only afterwards.
3. **Proxy-assisted estimates.** Combine the model's scores on many episodes with the
   reviewed sample to estimate each arm's success rate with a narrower interval
   ([PPI++](../stats/ppi.md)).

The primary analysis is unchanged: it always uses the human labels of the scheduled trials.

## Install

```console
$ pip install 'fieldtrial[rewards]'      # Python 3.12 or later; installs LeRobot and torch
```

The built-in scorers use LeRobot 0.6's reward models:

| `--model` | Score | Notes |
|---|---|---|
| `robometer` | Predicted progress at the episode's last frame, from up to 8 frames spread over the episode | Default checkpoint `lerobot/Robometer-4B` (`--pretrained` to change it) |
| `topreward` | Probability that the instruction was completed, `exp(log P("True"))`, from up to 16 frames | Zero-shot with a Qwen3-VL model; `--pretrained` for a TOPReward config |
| `package.module:factory` | Your own | `factory(camera=..., device=..., pretrained=...)` returns an object with `name`, `score(root, episode_index)` (0 to 1) and `close()` |

Both models need a GPU for reasonable speed (`--device cuda`, the default). fieldtrial's
own tests do not load them; they were written against LeRobot 0.6.1's source and have not
been run with real weights in fieldtrial's CI.

## 1. Agreement with the operators

Record trials with LeRobot, [link them](lerobot.md) to the dataset's episodes, then score
the dataset:

```console
$ fieldtrial link-episodes my-study user/eval_rollouts --yes
$ fieldtrial score-episodes my-study user/eval_rollouts --model robometer \
    --camera observation.images.top
Scored 80 episodes with robometer:lerobot/Robometer-4B (80 linked to trials); 58 suggested successes.
```

A score at or above the threshold (0.5, or `analysis.proxy.threshold`) is a suggested
success. After unblinding, the report's **Reward-model agreement** table compares the
suggestions with the operators' labels. The operators labelled the trials live, before any
score existed, so these labels are blind to the model.

## 2. Blind review of extra episodes

Score a dataset of extra rollouts of one arm, naming the arm by its blind code:

```console
$ fieldtrial score-episodes my-study user/overnight_k7 --model robometer --arm K7
$ fieldtrial review-sample my-study user/overnight_k7 --n 40
Drew 40 episodes for review.
```

The sample is drawn with the study's seed and recorded before anyone reviews, so it cannot
depend on the scores. Open the study in the console and choose **Review episodes**: the
page plays the episode's span of the dataset video and asks for a label. Only after the
label is saved does it show the model's suggestion; if the reviewer then changes their
mind, the change is kept with a reason, but agreement is computed from the first label.

## 3. Proxy-assisted estimates

Pre-register the secondary analysis before locking:

```yaml
analysis:
  primary:
    comparison: {treatment: q50, control: baseline}
  proxy: {threshold: 0.5}
```

For each scored dataset and arm, the report's **Proxy-assisted estimates** table combines
the reviewed sample (human labels) with every other scored episode (model scores only),
and shows the interval from the human labels alone next to it. With both arms scored from
the same source, it also shows their difference. See [PPI++](../stats/ppi.md) for the
method and when it is valid.

Scores from elsewhere, for example simulation (as in SureSim), can be imported:

```console
$ fieldtrial import-proxy my-study sim_scores.csv --source sim
```

The CSV has the columns `blind_code,score,label`; rows with a label (1 or 0) are the
labeled set and must be a random subset of their arm's rows.

## What it records

Everything is in the event log: `episode_score` (model, threshold, every score),
`review_sample`, `review_label`, `review_revision` (with the reason) and `proxy_import`.
Scores never change trial labels.
