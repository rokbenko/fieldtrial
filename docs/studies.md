# Running a study

A study is a folder. `study.yaml` holds the design you edit; locking it creates
`fieldtrial.db`, which from then on holds the frozen design, the randomized schedule, every
trial and an append-only event log.

```text
my-study/
├── study.yaml        # the design (editable; changes after locking need `amend`)
├── fieldtrial.db     # created by `lock`: design, schedule, trials, events
├── media/            # clips and photos attached to trials
└── reports/          # report.md and results.json
```

## 1. Design

```console
$ fieldtrial init my-study --template basic
$ fieldtrial validate my-study
$ fieldtrial plan my-study --baseline 0.76
```

Templates: `basic` (two arms, 40 starting positions), `checkpoint-ladder` (four checkpoints
with a pre-registered [ladder](guides/checkpoint-ladders.md)), `crossover-rounds` (two arms
on a task whose scene carries over) and `serving-sweep` (three serving settings across
objects and positions). `validate`
reports every problem with its line in `study.yaml`. `plan` previews the schedule and the
minimum detectable effect for the planned number of trials.

The `plan` MDE uses the independent-samples formula. Paired designs usually detect
somewhat smaller differences, so the number is conservative.

### Designs

| `design.type` | When | Primary analysis |
|---|---|---|
| `randomized_block` | Each trial starts from a reset scene | Exact McNemar (2 arms), Cochran's Q + Holm (more arms), CMH (replicates) |
| `crossover_rounds` | The scene carries over between trials (filling a tray) | [Crossover rounds](stats/crossover.md) |
| `single_arm` | One policy against a fixed bar | Exact binomial test against `threshold` |

**Crossover rounds.** Each arm runs a whole round: one pass over every condition, without
resetting the scene in between. Rounds come in cycles of two, one per arm, in a randomized
and balanced order. Set `limits.reset: carry_over` and the number of rounds:

```yaml
limits:
  reset: carry_over
design:
  type: crossover_rounds
  rounds: 16          # 8 cycles
  order: fixed        # the same condition order in every round
```

A round is the unit of analysis, so plan enough cycles: with 4 cycles, no result can reach
p < 0.05. Crossover designs compare exactly 2 arms and need `conditions.replicates: 1`. An
invalid trial is retried at the end of the same round.

**Stopping early.** A two-arm `randomized_block` study can plan interim looks:

```yaml
analysis:
  stopping: {rule: group_sequential, looks: 4, spending: obrien_fleming}
```

The study is analyzed after 25%, 50% and 75% of the blocks (or at the fractions in `at:`)
and stops if the [error-spending boundary](stats/sequential.md) is crossed. Run a look with
`fieldtrial interim my-study` or from the console when it is due. A look reports only
"continue" or "stop", so the study stays blinded. A stop cancels the remaining trials.

To check after every block instead of at a few planned looks, use the
[anytime-valid rule](stats/anytime.md):

```yaml
analysis:
  stopping: {rule: anytime}      # optional: min_blocks: 10
```

Nothing needs to be run by hand: each time a trial completes a block, fieldtrial applies the
pre-registered test and stops the study (cancelling the remaining trials) as soon as it
rejects. It stops early on large differences, but has less power than a fixed or
group-sequential design for small ones.

**Selecting the best of several arms.** With three or more arms, for example serving
configurations, `analysis.selection` replaces the comparison:

```yaml
analysis:
  primary: {}
  selection: {rule: elimination, delta: 0.05}
```

After every complete block, an arm that another arm beats with confidence is
[dropped](stats/selection.md) and its remaining trials are cancelled; the study stops when
one arm remains. The console names dropped arms by blind code only. The `best-arm`
template sets this up for four serving configurations.

**Reward-model estimates.** To report [proxy-assisted estimates](stats/ppi.md) from
reward-model scores, pre-register them before locking:

```yaml
analysis:
  proxy: {threshold: 0.5}   # scores from 0.5 up count as suggested successes
```

See [reward models and blind review](guides/reward-models.md) for the workflow.

## 2. Lock

```console
$ fieldtrial lock my-study
Locked: design 2079dce2818b, 80 trials scheduled
```

Locking stores the design YAML, its hash and the randomized schedule. Each block is one
condition × replicate; every arm runs once per block, in a Williams-balanced random order.
The same seed gives the same schedule on every platform and numpy version.

From now on the design comes from the database, not from `study.yaml`. To change it, edit
`study.yaml` and run:

```console
$ fieldtrial amend my-study --reason "two more start positions"
```

Completed trials are kept. Pending trials the new design no longer contains are removed,
and new ones are appended. The seed cannot be amended, and neither can the rounds of a
crossover design. Every report lists amendments as
deviations, with the reason and the old and new design hash.

## 3. Run trials

Run trials at the robot with the [operator console](console.md) (`fieldtrial serve`),
from your own runtime through the [REST API](api.md), by simulation, or by importing a
CSV.

**Simulate** (sim runner and auto-operator, for trying things out):

```console
$ fieldtrial simulate my-study --rates baseline=0.76,q50=0.90 --seed 1
Simulated 80 completed and 0 invalid trials in 2 sessions
```

`--invalid-rate 0.05` voids some trials as simulated robot faults, and `--max-trials 20`
stops early.

**Import** trials you recorded elsewhere:

```console
$ fieldtrial import my-study results.csv --map arm=policy,success=ok
```

Required columns are `condition` (for example `slot=3`), `arm`, and either `success`
(yes/no, true/false, 1/0) or `stage` (a stage id, an index, or empty for no stage). Optional
columns are `duration_s`, `termination`, `failure_tags` (separated by `;`), `operator`,
`rig`, `session`, `status` (`invalid` imports a voided trial), `invalid_reason` and `notes`.
`--map field=column` renames columns.

Each row fills the next pending slot with the same condition and arm. Every row is checked
before anything is written: if one row cannot be placed, nothing is imported and every
problem is listed.

**Invalid trials** (a robot fault, a setup error) are never deleted. The slot is voided and
a replacement is scheduled at the end of its block, so the design stays balanced. Reports
count invalid trials per arm and flag an imbalance.

## 4. Blinding

With `blinding: operator`, the operator sees only blind codes, such as `N4`, and
`fieldtrial status` shows progress without per-arm results. `analyze` and `report` refuse to
run until you unblind:

```console
$ fieldtrial unblind my-study
```

Unblinding is logged. If trials were still pending, every report flags it.

!!! warning "Manual mode only half-blinds"
    With the `manual` runner, the operator loads the checkpoint, so they can know which
    arm is running. For real blinding, use a [switching runner](guides/runners.md).

## 5. Analyze

```console
$ fieldtrial analyze my-study           # summary in the terminal
$ fieldtrial analyze my-study --json    # the full Results model
$ fieldtrial report my-study            # reports/report.html, with charts
$ fieldtrial report my-study --format md
$ fieldtrial report my-study --format json
$ fieldtrial export my-study --format csv
```

See [Analysis and reports](analysis.md) for what the analysis contains.
