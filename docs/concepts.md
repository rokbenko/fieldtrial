# Concepts

A fieldtrial study answers one pre-registered question, such as "does queue threshold 50
succeed more often than 30?", with a design that keeps the comparison fair. This page
explains the parts of `study.yaml` and why they exist.

## Arms

An **arm** is one thing being evaluated: a policy checkpoint, a serving configuration, or
both. Each arm has an `id`, an optional `label`, a `runner`, and two free-form
dictionaries, `policy` and `serving`, that are passed to the runner unchanged.

```yaml
arms:
  - id: baseline
    label: "queue 30"
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference.type: rtc, inference.queue_threshold: 30}
  - id: q50
    label: "queue 50"
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference.type: rtc, inference.queue_threshold: 50}
```

The primary analysis compares a **treatment** arm with a **control** arm
(`analysis.primary.comparison`). With more than two arms, every arm is compared with the
control, with a multiplicity adjustment.

## Conditions and blocks

A **condition** is one starting setup: an object position, a lighting setting, an object.
Factors expand into the full list of conditions (the cartesian product), and
`replicates` repeats each condition.

```yaml
conditions:
  factors:
    slot: {range: [1, 40]}
    object: {values: [cup, bowl]}   # 80 conditions
  replicates: 1
```

A **block** is one condition × replicate. In a `randomized_block` design every arm runs
once in every block, in a random order balanced across arms (Williams-style), and the
order of blocks is random too. Each arm therefore meets the same easy and hard
conditions, at similar times of day. This is what makes a **paired** analysis possible:
it compares arms within blocks, so differences between conditions cancel out.

## Rubric and stages

The **rubric** defines what you record. **Stages** are ordered milestones; reaching a
stage implies reaching every earlier one. **Success** is reaching the `success` stage.

```yaml
rubric:
  stages:
    - {id: lift,     label: "Lifted the correct part"}
    - {id: handover, label: "Clean handover"}
    - {id: inserted, label: "Inserted, but tilted"}
    - {id: clean,    label: "Clean insertion"}
  success: clean
  failure_tags: [dropped, missed_grasp, collision]
```

The operator records the furthest stage reached, so the report can show *where* arms
fail (a stage funnel), not only how often they succeed. Failure tags say *how*.

## Blinding

With `blinding: operator`, the console shows each arm only by a **blind code** such as
`N4`, and per-arm results stay hidden until someone unblinds the study (which is logged).
An operator who knows which arm is running may, without meaning to, reset the scene more
carefully or call a borderline outcome differently.

With the `manual` runner, blinding is partial: the operator still loads the checkpoint.
Real blinding needs a runner that switches arms itself (planned for v0.2).

## Pre-registration and locking

`fieldtrial lock` freezes the design before the first trial: the YAML, its hash, and the
randomized schedule are stored in the study's database. The analysis plan follows from
the locked design. The primary test is chosen by the design, never after seeing the data
(see [Analysis and reports](analysis.md)).

Designs can still change: `fieldtrial amend --reason "..."` applies edits as a logged
amendment, and every report lists it as a deviation. Cosmetic fields (title, labels,
descriptions) do not change the hash.

## Invalid trials

Sometimes a trial cannot count: the robot faults, a camera falls, the wrong part was in
the tray. Mark it **invalid** with a reason. Invalid trials are never deleted: they are
counted per arm in every report, a check flags arms with unusually many of them, and the
slot is rescheduled at the end of its block, so the design stays balanced.

Invalid is not a softer word for failure. A policy that drops the part has failed; a
gripper that loses power mid-trial has not.

## Sessions

A **session** is one operator on one rig for one stretch of work, started with the rig
checklist. Reports test whether each arm's success rate changed across sessions and
operators. A change usually means something on the rig changed.

## Seeds

Every random choice (schedule, blind codes, bootstrap intervals, simulations) comes from
the study's `seed`, through a generator that gives the same numbers on every platform and
numpy version. The same `study.yaml` always gives the same schedule.
