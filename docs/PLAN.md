# fieldtrial — build plan and kickoff prompt for Claude Code

> This document is both the kickoff prompt and the project's source of truth.
> Paste it into Claude Code in an empty folder, or save it as `docs/PLAN.md` and tell Claude Code to read it.

---

## 0. How we work

You are the lead engineer building **fieldtrial**, an open-source Python framework for statistically rigorous, real-world evaluation of robot policies. I am the maintainer and I review at every milestone.

1. **Read this whole document first.** If `docs/PLAN.md` does not exist, create it with this document's full content, verbatim. It is the source of truth: when we change a decision, update it and add a line to the Decision log (Appendix B).
2. **Create `CLAUDE.md`** from §21. This is the project memory for future sessions.
3. **Plan before you code.** Give me a short plan (one page at most) for Milestones 0 and 1. Ask at most 5 questions, and only blocking ones; every decision here has a default. Then wait for my OK.
4. **Work one milestone at a time** (§20).
   - Take small steps.
   - Write tests first for anything statistical.
   - Run lint, type checks and tests before every commit.
   - Use Conventional Commits.
   - Push only your working branch on `rokbenko/fieldtrial`. Opening PRs, pushing tags and pushing other branches need my OK.
5. **Stop at the end of each milestone.** Summarize what you built, give me the exact commands to try it, list open questions, and say what comes next. Do not start the next milestone without my OK.
6. **Verify third-party APIs.** Everything in §13 about LeRobot and openpi was checked against LeRobot 0.6.1 and openpi-client 0.1.2 on 2026-10-01. Re-check against the installed source before depending on it. Never write an integration from memory.
7. **Keep scope tight.** Anything not in the current milestone goes into `docs/ROADMAP.md` instead of the code.

---

## 1. Why this exists

Fine-tuning a vision-language-action (VLA) policy is now routine: LeRobot, openpi and Isaac-GR00T all ship recipes. What is still hard is knowing whether the fine-tuned policy actually got better. Real-world evaluation has three problems:
- **Slow:** a human resets the scene for every rollout.
- **Noisy:** with 40 rollouts the 95% interval is about ±15 percentage points.
- **Ad hoc:** it usually lives in spreadsheets.

Dream Machines' π0.5 fine-tuning study (https://dream-machines.eu/blog/pi05-fine-tuning, September 2026) shows the problem concretely:
- Learning rate, batch size, action space and image augmentation all landed within the noise of a 40-rollout evaluation.
- Validation loss and validation action error did not predict which checkpoint was better.
- A hardware confound (gripper plasticity) forced them to rerun every evaluation.
- They skipped an obvious follow-up experiment because 40 rollouts could not resolve it.

Re-analyzing their published counts makes the point. Their headline result, that one clean hour lifted success from 76% to 90% (36/40 vs 91/120), has a 95% CI for the difference of −0.5 to +24.5 pp (Boschloo p ≈ 0.056). That is suggestive, not conclusive.

They built rubrics, progress stages, confidence intervals and checkpoint ladders by hand. Every team that deploys a policy rebuilds the same tools. fieldtrial packages them:

- **Before an evaluation:** How many rollouts do I need? What is the smallest difference this evaluation can detect?
- **During:** a randomized, blinded schedule, plus a phone-friendly operator console that logs outcome, progress stage and failure tags for each rollout.
- **After:** correct statistics (exact tests, paired designs, multiplicity control), honest wording, and a shareable report.

**Pitch:** LeRobot trains the policy; fieldtrial tells you whether it actually got better.

**fieldtrial is not:**
- a training framework
- a simulation benchmark (LeRobot's `lerobot-eval` covers that)
- a robot driver
- a labeling platform
- a cloud service

It complements LeRobot and openpi and never forks them.

---

## 2. Identity

- **Name:** `fieldtrial`. It is the PyPI name, the import name and the CLI command. It was free on PyPI as of 2026-10-01 and is reserved with a placeholder release, `0.1.0.dev0`.
- **Story (README footnote):** named after agricultural field trials. At Rothamsted in the 1920s, R. A. Fisher developed randomized block designs for field trials, and those are the designs fieldtrial uses to compare robot policies.
- **Tagline:** "Find out whether your robot policy actually got better."
- **License:** Apache-2.0.
  - Author and copyright holder: Rok Benko. Project metadata carries no email address.
  - Don't invent URLs. The repository is `https://github.com/rokbenko/fieldtrial`.
- **Telemetry:** none, ever.

---

## 3. Principles (non-negotiable)

1. **Correctness first.** Every statistical function has golden-value tests against an independent implementation, plus property-based tests. A wrong p-value is worse than no p-value.
2. **Pre-registration.** The design is fixed and hashed before the first trial. The design covers arms, conditions, the primary comparison, the test, α and the stopping rule. Any change after locking is a logged amendment, and it appears in the report.
3. **Honest language.** Reports never call an arm "better" unless the pre-registered primary test rejects. Every report states what the study could and could not resolve, as a minimum detectable effect (MDE).
4. **Works with any stack.** Manual mode comes first: fieldtrial schedules and records, and you run the robot however you like. Integrations are adapters.
5. **Local-first and offline.** One folder per study, SQLite storage, vendored web assets. It must work on a factory LAN with no internet.
6. **Lightweight core.** `fieldtrial.stats` depends only on numpy and scipy. Nothing in the core imports torch, lerobot or openpi.
7. **Reproducible.** All randomness comes from the study seed. Reports embed software versions, the design hash and policy fingerprints.

---

## 4. Users and the core workflow

**Primary user:** an ML or robotics engineer on a small team or in a lab, with 1–3 real robot setups (rigs), fine-tuning open VLAs (π0.5, GR00T, SmolVLA and similar) for a specific task.

**Secondary user:** the operator who runs the rollouts, often a technician, using a phone or tablet next to the robot.

**Workflow:**
1. `fieldtrial init my-eval` creates `study.yaml` from a template.
2. Edit the study:
   - **arms:** checkpoints and serving configurations
   - **conditions:** start positions, objects
   - **rubric:** progress stages and failure tags
   - **primary comparison**
3. `fieldtrial plan my-eval` validates the study, previews the schedule, and shows power and MDE for the planned number of trials. Example: "40 per arm detects improvements of ≥21 pp from a 76% baseline."
4. `fieldtrial lock my-eval` freezes the design.
5. `fieldtrial serve my-eval` prints a URL and QR code for the console. The operator runs each trial: Start, robot runs, Stop, tap the stage reached and failure tags, next.
6. `fieldtrial report my-eval` produces an HTML/Markdown/JSON report. It includes:
   - confidence intervals and tests
   - stage funnels
   - a condition-by-arm heatmap
   - session drift checks
   - the invalid-trial breakdown

The calculator commands also work on their own, with no study at all: `fieldtrial ci 36/40`, `fieldtrial compare 74/80 91/120`, `fieldtrial power --p1 0.76 --p2 0.90`. They are how most people will discover the tool.

---

## 5. Scope

**v0.1 (Milestones 0–4):**
- statistics library and calculator CLI
- study design:
  - YAML spec and validation
  - randomized complete block schedule
  - blinding codes
  - lock and amend
- SQLite store with an append-only event log
- CSV import and export
- analysis engine
- web operator console with manual mode and a simulation runner
- REST API and Python client
- HTML, Markdown and JSON reports
- documentation site
- re-analysis of Dream Machines' published results as an example
- PyPI release

**v0.2:**
- command-template runner: spawns your rollout command per arm, which makes real blinding possible
- openpi router: blinded A/B for websocket policy servers
- evaluation-camera capture (`[capture]` extra)
- links between trials and LeRobot dataset episodes
- checkpoint ladders: trend test and plateau detection
- crossover-rounds design for tasks where scene state carries over between trials
- rig drift check against reference images
- group-sequential stopping

**v0.3:**
- in-process LeRobot runner
- reward-model pre-labeling (Robometer/TOPReward via `lerobot.rewards`), with human confirmation and agreement tracking
- proxy-assisted intervals using prediction-powered inference (compare SureSim, arXiv:2510.04354)
- serving-config sweeps with best-arm identification
- near-optimal sequential comparison: evaluate Snyder et al., RSS 2025, arXiv:2503.10966, and check its code and license before porting anything

**Non-goals:**
- training
- simulation benchmarks
- fleet monitoring
- multi-tenant authentication
- cloud hosting
- any single-page-app or Node build step

---

## 6. Architecture

```
 CLI (typer)        Web console (FastAPI + Jinja2 + HTMX)        REST API /api/v1  +  fieldtrial.client
      │                              │                                     │
      └──────────────────────────────┼─────────────────────────────────────┘
                                     ▼
                services/  (study, session, trial, analysis, events)
          ┌──────────────┬───────────┴──────────┬───────────────────────┐
          ▼              ▼                      ▼                       ▼
      design/         store/               analysis/ ───────────►    stats/
  (YAML → models,   (SQLAlchemy 2 +       report/ (Jinja2 +       (numpy + scipy only)
   schedule, hash)   Alembic, event log)   matplotlib SVG)
          ▲
      runners/  (manual, sim  |  later: command, openpi_router, lerobot)
```

**Dependency rules.** Enforce them with an `import-linter` contract that runs in CI.
- `stats` imports no other fieldtrial module, and only numpy and scipy.
- `design` may import `stats`, but only for power and MDE.
- CLI, web and API call `services`. They never touch the DB session or SQL directly.
- `runners` sit behind a Protocol. Optional dependencies are lazy-imported inside the adapters and guarded by extras. Nothing imports lerobot, openpi or torch at module import time.
- Every write goes through a service, in one transaction that also appends to the event log.

---

## 7. Tech stack

**Python:** ≥3.11.
- openpi's environment uses 3.11.
- LeRobot ≥0.5 requires ≥3.12, so the `lerobot` extra carries a `python_version >= "3.12"` marker.
- CI runs 3.11, 3.12 and 3.13.

**Packaging:** uv for environments and the lockfile; hatchling plus hatch-vcs as the build backend (version comes from git tags).

**Core dependencies:**
- numpy, scipy
- pydantic v2, pyyaml
- typer, rich
- SQLAlchemy 2.x, alembic (with `render_as_batch=True` for SQLite)
- fastapi, uvicorn[standard], jinja2, sse-starlette
- matplotlib
- qrcode

Each core dependency is added to `pyproject.toml` in the milestone that first imports it.

**Extras** (each is declared in the milestone whose code first uses it, so `openpi`, `capture`, `lerobot` and `rewards` arrive in v0.2 or later):

| Extra | Contents |
|---|---|
| `openpi` | websockets |
| `capture` | opencv-python-headless, av |
| `lerobot` | `lerobot>=0.6,<0.7` |
| `rewards` | `lerobot[robometer,topreward]` |
| `docs` | mkdocs-material, `mkdocstrings[python]` |
| `dev` | pytest, pytest-cov, hypothesis, statsmodels (test reference only, never imported by `src`), ruff, mypy, pre-commit, httpx, import-linter |

**Frontend:** server-rendered Jinja2, HTMX 2, Alpine.js 3, and one small hand-written CSS file. Vendor the JS files into `src/fieldtrial/web/static/vendor/`; no CDN and no Node.

**Quality and docs:**
- **Lint and format:** ruff.
- **Types:** mypy in strict mode for the whole package. This is stricter than strict-for-`stats`-and-`design`, but mypy can't enable `strict` per module.
- **Docs:** MkDocs Material with mkdocstrings, deployed to GitHub Pages.

---

## 8. Repository layout

```
fieldtrial/
├── pyproject.toml  uv.lock  README.md  LICENSE  CHANGELOG.md
├── CLAUDE.md  CONTRIBUTING.md  CODE_OF_CONDUCT.md  SECURITY.md  mkdocs.yml
├── docs/            PLAN.md  ROADMAP.md  index.md  quickstart.md  concepts/  guides/  stats/  reference/
├── examples/
│   ├── dream-machines-pi05/   published_counts.csv  reanalysis.py  README.md
│   └── templates/             basic/  checkpoint-ladder/  serving-sweep/   (study.yaml each)
├── src/fieldtrial/
│   ├── __init__.py            version + curated public API
│   ├── stats/                 proportions.py compare.py paired.py power.py multiplicity.py
│   │                          bayes.py ordinal.py timing.py _types.py
│   ├── design/                models.py loader.py schedule.py blinding.py hashing.py
│   ├── store/                 db.py models.py repo.py migrations/
│   ├── services/              study.py session.py trial.py analysis.py events.py
│   ├── analysis/              engine.py results.py wording.py
│   ├── report/                html.py markdown.py charts.py templates/
│   ├── runners/               base.py manual.py sim.py    (later: command.py openpi_router.py lerobot.py)
│   ├── web/                   app.py routes/ templates/ static/{app.css,vendor/}
│   ├── api/                   v1.py schemas.py
│   ├── io/                    csv.py jsonl.py
│   ├── client.py              thin REST client for custom robot runtimes
│   └── cli/                   main.py calc.py study.py serve.py
└── tests/                     mirrors src/ (+ e2e/)
```

---

## 9. Domain model

**Concepts:**
- **Study:** one pre-registered comparison. It is a folder containing `study.yaml`, `fieldtrial.db`, `media/` and `reports/`.
- **Arm:** what is being evaluated: a policy reference, a serving configuration and a runner. Each arm gets a blind code such as `K7`.
- **Condition:** an initial setup the robot is tested in, for example `slot=17` or `object=blue_cup`. Conditions are the blocks of the design.
- **Schedule:** the ordered list of (condition, arm) slots generated from the seed.
- **Session:** one continuous working period by one operator on one rig. It records start and end times, the rig check and environment notes.
- **Trial:** one rollout attempt for a schedule slot.
  - The outcome is the furthest progress stage reached; success means reaching the success stage.
  - A trial also records the termination reason, failure tags, duration, notes and media links.
  - A trial can be marked invalid (robot fault, setup error) with a reason. The slot is then rescheduled. Invalid trials are never silently dropped.
- **Event:** the append-only audit log (design locked, amendment, trial started, completed, invalidated or edited, unblinding, …).

**Tables.**
- **Conventions:** SQLAlchemy 2 typed ORM, UUIDv7 string primary keys from a small helper, UTC timestamps, SQLite in WAL mode with foreign keys enabled.
- **study:** `id, name, title, design_yaml, design_hash, status[draft|locked|running|closed], created_at, locked_at, closed_at, fieldtrial_version`
- **arm:** `id, study_id, key, blind_code, label, runner, policy_ref JSON, serving JSON, policy_fingerprint`
- **condition:** `id, study_id, key, factors JSON, position`
- **schedule_slot:** `id, study_id, seq, block, replicate, condition_id, arm_id, origin[planned|rescheduled], status[pending|done|void]`
- **session:** `id, study_id, operator, rig, started_at, ended_at, rig_check JSON, environment JSON, software JSON, notes`
- **trial:**
  - `id, study_id, slot_id, session_id, attempt, status[running|completed|invalid]`
  - `started_at, ended_at, duration_s`
  - `stage_index` (−1 = none), `success`
  - `termination[success|timeout|stuck|operator_stop|robot_fault|safety_stop|other]`
  - `failure_tags JSON, notes, invalid_reason, media JSON, auto_labels JSON, version`
- **event:** `id, study_id, ts, kind, actor, payload JSON`

**Policy fingerprint:**
- for a Hugging Face model, the revision hash
- for a local checkpoint, a sha256 of the weight files, cached by size and mtime

**Concurrency:** use a `version` column for optimistic concurrency on trials, and idempotency keys on console POSTs (phones double-tap).

---

## 10. `study.yaml`

```yaml
fieldtrial: 1                    # schema version
name: rtc-queue-sweep
title: "RTC queue threshold on the 21h pi0.5 checkpoint"
task:
  instruction: "Pick the part, hand it over, insert it into the tray"   # passed to the policy if the runner needs it
  description: |
    Bimanual pick, handover, insertion. Top layer of the box only.

rubric:
  stages:                        # ordered; reaching stage k implies every earlier stage
    - {id: lift,     label: "Lifted the correct part"}
    - {id: handover, label: "Clean handover, held at least 1 s"}
    - {id: inserted, label: "Inserted, but tilted or misoriented"}
    - {id: clean,    label: "Clean insertion, correct orientation"}
  success: clean                 # success = reached this stage
  failure_tags: [dropped, missed_grasp, wrong_part, collision, stuck, order_violation]
  calibration_media: rubric/     # optional: one example clip or photo per stage, shown in the console
  rig_checklist: ["Gripper pads clean", "Lamp on", "Tray empty and seated"]

limits:
  timeout_s: 45
  no_progress_s: 10              # shown as a hint; the operator calls "stuck"
  reset: independent             # independent | carry_over (carry_over needs design.type: crossover_rounds, v0.2)

arms:
  - id: baseline
    label: "queue 30"
    runner: manual
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference.type: rtc, inference.queue_threshold: 30}       # passed through to the runner
  - id: q50
    label: "queue 50"
    runner: manual
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference.type: rtc, inference.queue_threshold: 50}

conditions:
  factors:
    slot: {range: [1, 40]}       # cartesian product of all factors = the condition list
  replicates: 1

design:
  type: randomized_block         # v0.1: randomized_block | single_arm   (v0.2: crossover_rounds)
  order: random                  # random | fixed (when physical constraints dictate the condition order)
  blinding: operator             # none | operator
  seed: 20261001

analysis:
  primary:
    metric: success
    comparison: {treatment: q50, control: baseline}
    alternative: two-sided
    alpha: 0.05
  interval: wilson
  multiplicity: holm             # only used when there are several comparisons
  secondary: [stage_reached, time_to_success]
  stopping: {rule: fixed}        # v0.2: group_sequential
```

**Validation rules** (pydantic, with friendly errors that point to the YAML path):
- Ids and stage ids are unique, and the success stage exists.
- Comparative designs need at least 2 arms. `single_arm` takes 1 arm and an optional threshold test, for example "is success ≥ 0.90?".
- Factors expand to a finite condition list: the cartesian product, with inclusive `range`. Warn above 500 conditions.
- The primary comparison references existing arms; 0 < α ≤ 0.2.
- Unknown keys are errors (`extra="forbid"`). The exceptions are `policy` and `serving`, which are runner-specific passthrough dicts.
- The design hash is the sha256 of the normalized model dump (sorted keys), excluding cosmetic fields: `title`, `description`, `label`, `instruction`.

---

## 11. Design and scheduling

**`randomized_block` (default):**
- Each block is one condition × replicate. Within a block, every arm runs once, in random order.
- Block order is random unless `order: fixed`.
- Balance within-block positions across arms where possible (Williams-style), so that time-of-day and carryover effects average out.

**Determinism:** use `numpy.random.Generator(PCG64(seed))`. The same seed and design must give an identical schedule on every platform; snapshot-test this.

**Blinding (`blinding: operator`):**
- The console shows only blind codes.
- Arm identities and per-arm interim results stay hidden until `fieldtrial unblind`. Unblinding is logged, and the report flags unblinding before completion.
- Docs must say plainly that manual mode only half-blinds: the operator still loads the checkpoint. Real blinding needs a switching runner (v0.2).

**Invalid trials:** the slot is rescheduled at the end of the current block by default (option: end of study), keeping the design balanced. Invalid counts are reported per arm.

**Sessions:** a session can stop and resume at any point; resuming continues the schedule.

**`crossover_rounds` (v0.2):** for tasks whose scene evolves across trials, such as filling a tray.
- A round is a full pass over the conditions with one arm.
- Arm order across rounds is counterbalanced (ABBA…).
- Analysis is done at the round level.

---

## 12. Statistics (`fieldtrial.stats`)

**Conventions:** pure, typed functions returning frozen dataclasses (`Interval`, `ProportionEstimate`, `ComparisonResult`, `TestResult`, `PowerResult`). Every docstring gives the formula and a reference. Defaults: 95% intervals, two-sided tests, α = 0.05. Raise `ValueError` with clear messages on invalid input (n = 0, k > n, level outside (0, 1)).

**One arm:**
- `proportion_ci(k, n, *, level=0.95, method="wilson")`
  - Wilson is the default.
  - Clopper–Pearson (via beta quantiles), Jeffreys and Agresti–Coull are also available.
  - Clip bounds to [0, 1].
- `test_vs_threshold(k, n, p0, *, alternative)`: exact binomial test (`scipy.stats.binomtest`).

**Two independent arms:** `compare_independent(k1, n1, k2, n2)` returns:
- the difference p1 − p2 with a Newcombe hybrid-score CI (method 10)
- the primary test: Boschloo exact (`scipy.stats.boschloo_exact`), which is uniformly more powerful than Fisher's exact test
- Fisher's exact test alongside, for comparability with papers
- the odds ratio (optional)

**Paired (complete blocks):**
- **2 arms:** exact McNemar on the discordant pairs (b, c), plus a Tango score CI for the paired difference. Validate Tango against an independent implementation. If no Python reference exists, write an R snippet using `PropCIs::scoreci.mp` and ask me to run it and paste the results.
- **More than 2 arms:** Cochran's Q as the omnibus test, then pairwise McNemar with Holm correction.
- **Replicates > 1 per condition:** Cochran–Mantel–Haenszel test, stratified by condition.

**Several comparisons:** each arm against the control (default) or all pairs. Holm correction by default; Bonferroni and Benjamini–Hochberg are also available.

**Planning** (every planning function supports unequal allocation via `ratio = n2/n1`, because real studies often compare 40 new trials against 120 baseline trials):
- `sample_size(p1, p2, *, alpha, power, method, ratio=1.0)`:
  - pooled-z (default)
  - Fleiss continuity-corrected
  - arcsine (Cohen's h)
- `power(...)`: closed forms, plus exact power for Boschloo and McNemar. It is computed by enumerating every possible outcome, so it has no Monte Carlo error. Seeded simulation remains as a cross-check.
- `mde(p_baseline, n, *, alpha, power, direction)`: the minimum detectable effect.

**Bayesian (descriptive only, never the primary test):**
- `prob_superiority(ka, na, kb, nb, prior=(1, 1))` gives P(p_b > p_a), computed by numerical integration over the Beta posteriors.
- Credible intervals.

**Progress stages:**
- the per-arm distribution of the furthest stage reached
- a stage funnel with conditional conversion rates and Wilson CIs
- a Brunner–Munzel test on the stage index (`scipy.stats.brunnermunzel`) as a secondary comparison

**Timing:**
- a cumulative success curve per arm: the fraction of all trials that had succeeded by time t. Every trial is observed until it ends, so there is no censoring.
- median time-to-success among successful trials, with a seeded bootstrap CI

**Drift and validity flags:**
- **Session drift:** per-session and per-operator success rates for each arm, with a homogeneity test (Fisher or chi-square). Flag "results changed across sessions"; this is the kind of drift the gripper confound in the Dream Machines study caused.
- **Invalid-trial imbalance:** flag when one arm has noticeably more invalid trials than another.

### Golden values (must match)

Computed with scipy 1.17.1 and statsmodels 0.15.0 on 2026-10-01. Absolute tolerance is 1e-4 unless noted. Where statsmodels has the same method, the test should also re-derive the value live.

**One-sample 95% intervals**

| k/n | Wilson | Clopper–Pearson | Jeffreys |
|---|---|---|---|
| 91/120 | [0.6745, 0.8261] | [0.6717, 0.8318] | [0.6762, 0.8282] |
| 36/40 | [0.7695, 0.9604] | [0.7634, 0.9721] | [0.7796, 0.9653] |
| 13/40 | [0.2008, 0.4798] | [0.1857, 0.4913] | [0.1960, 0.4784] |
| 39/40 | [0.8712, 0.9956] | [0.8684, 0.9994] | [0.8891, 0.9973] |
| 0/40 | [0.0000, 0.0876] | [0.0000, 0.0881] | (skip; edge-case conventions differ) |
| 40/40 | [0.9124, 1.0000] | [0.9119, 1.0000] | (skip) |

- Agresti–Coull for 36/40: [0.7638, 0.9661].
- Wilson half-width at 20/40: 14.8 pp, which is the "±15 pp at n = 40" rule of thumb.

**Two independent arms** (difference = p1 − p2)

| Comparison | Difference | Newcombe 95% | Fisher p | Boschloo p |
|---|---|---|---|---|
| 50/80 vs 13/40 | +0.3000 | [+0.1104, +0.4582] | 0.00337 | 0.00228 |
| 36/40 vs 91/120 | +0.1417 | [−0.0054, +0.2450] | 0.07060 | 0.05574 |
| 74/80 vs 91/120 | +0.1667 | [+0.0625, +0.2596] | 0.00223 | 0.00196 |

The Dream Machines post itself reports Fisher p = 0.0034 for the first row, which makes a good sanity test. Boschloo values are stable across `scipy` grid sizes 32, 64 and 128. `scipy.stats.boschloo_exact` treats each column as one arm, so the table is `[[k1, k2], [n1 - k1, n2 - k2]]`.

**Exact McNemar test (two-sided)**

| Discordant pairs (b, c) | p |
|---|---|
| (10, 2) | 0.03857 |
| (7, 1) | 0.07031 |
| (5, 5) | 1.0 |

**Sample size per arm** (α = 0.05 two-sided, power 0.80). Test the raw float with relative tolerance 1e-6; the CLI prints the ceiling.

| p1 → p2 | pooled-z | Fleiss continuity-corrected | arcsine |
|---|---|---|---|
| 0.76 → 0.90 | 111.8216 (→112) | 125.7015 (→126) | 108.46 (→109) |
| 0.76 → 0.93 | 69.9517 (→70) | 81.2908 (→82) | 65.80 (→66) |
| 0.30 → 0.60 | 41.9703 (→42) | 48.4074 (→49) | 41.79 (→42) |

**Minimum detectable effect** (pooled-z, α 0.05 two-sided, power 0.80)

| Baseline, n per arm | Detectable increase | Detectable decrease |
|---|---|---|
| 0.76, n = 40 | to 0.9707 (+21.1 pp) | to 0.4588 (−30.1 pp) |
| 0.76, n = 120 | to 0.8958 (+13.6 pp) | to 0.5915 (−16.8 pp) |
| 0.50, n = 40 | to 0.7949 (+29.5 pp) | to 0.2051 (−29.5 pp) |

**Holm correction:** the Dream Machines serving-settings sweep, each arm against the baseline 91/120, using Fisher two-sided raw p-values.

| Arm | Count | Raw p | Holm-adjusted p | Reject at 0.05? |
|---|---|---|---|---|
| sync | 21/40 | 0.00893 | 0.05361 | no |
| R25-B50 | 20/40 | 0.00302 | 0.02113 | yes |
| R50-B50 | 22/40 | 0.01624 | 0.08118 | no |
| R50-B1 | 32/40 | 0.66934 | 1.0 | no |
| R50-B16 | 35/40 | 0.17923 | 0.60959 | no |
| R50-B20 | 74/80 | 0.00223 | 0.01785 | yes |
| R50-B25 | 68/80 | 0.15240 | 0.60959 | no |
| R50-B25 compiled | 29/40 | 0.67743 | 1.0 | no |

**Bayesian P(p_B > p_A)** with Beta(1, 1) priors

| A | B | P(p_B > p_A) |
|---|---|---|
| 91/120 | 74/80 | 0.99902 |
| 91/120 | 36/40 | 0.97068 |
| 13/40 | 50/80 | 0.99900 |
| 91/120 | 57/80 | 0.23101 |

### Statistical property tests (mark them `slow`, run nightly in CI)

- **Wilson coverage** at p ∈ {0.5, 0.75, 0.9} × n ∈ {20, 40, 80}: within 0.92–0.98, because Wilson coverage oscillates around the nominal level.
- **Type I error** of Boschloo and McNemar under the null: at most α + 3·SE.
- **MDE round trip:** power at the MDE ≈ 0.80.
- **Schedule invariants:** the schedule's balance properties hold.

---

## 13. Runners and integrations

```python
class Runner(Protocol):
    capabilities: RunnerCapabilities      # can_switch_arms, can_stop, records_media, reports_outcome
    def prepare(self, arm: ArmSpec) -> None: ...         # load or switch policy; may block (console shows "loading")
    def start(self, trial: TrialContext) -> None: ...
    def stop(self, reason: Termination) -> RunArtifacts: ...
    def status(self) -> RunnerStatus: ...
    def close(self) -> None: ...
```

### `manual` (v0.1)

No robot control. The operator runs the policy themselves. The console shows which arm to load (by blind code) and the reset instructions.

### `sim` (v0.1)

Produces synthetic outcomes from configured true success rates, stage probabilities and duration distributions, all seeded. It powers the tests, the docs, and `fieldtrial demo`. It has an optional auto-operator that fills a whole study in seconds.

### `command` (v0.2)

Runs a user-supplied command template per arm, for example:

```
lerobot-rollout --policy.path={policy.path} --robot.type=... --inference.type=rtc --inference.queue_threshold={serving[inference.queue_threshold]}
```

- **When to launch:** per trial, or per block. Loading a 3B-parameter model for every trial is slow; measure it and recommend per-block launching.
- **Stopping:** send SIGINT.
- **Logs:** captured per trial.
- **Environment variables:** `FIELDTRIAL_TRIAL_ID` and `FIELDTRIAL_ARM` (the blind code only).

### `openpi_router` (v0.2)

A websocket proxy that makes blinded A/B testing work with openpi-style policy servers. The protocol, as verified in openpi-client 0.1.2:
- The client connects, with an optional `Authorization: Api-Key …` header.
- The server first sends one msgpack metadata message.
- After that, each request is a binary msgpack-numpy observation frame, and each reply is a binary action frame. A text frame means a server error.

How the router works:
- **Upstreams:** it keeps one upstream connection per arm.
- **Forwarding:** it forwards frames verbatim to the active arm without decoding them, so it only depends on `websockets`.
- **Latency:** it records per-request latency per arm.
- **Metadata check:** it refuses to start if the arms' metadata differ.
- **Client side:** the robot client points at the router and never changes.

### `lerobot` (v0.3; spike in v0.2)

Verified against LeRobot 0.6.1, which requires Python ≥3.12.

**What exists today:**
- **Rollout strategies:** `lerobot-rollout` supports `base|sentry|highlight|dagger|episodic`. They are dispatched by a hard-coded if-chain in `lerobot/rollout/strategies/factory.py::create_strategy`, so we cannot register a new strategy through the CLI.
- **Plugin discovery:** LeRobot's third-party plugin discovery only imports packages named `lerobot_robot_*`, `lerobot_camera_*`, `lerobot_teleoperator_*`, `lerobot_policy_*` or `lerobot_env_*`.

**Plan:**
- Reuse `from lerobot.rollout import RolloutConfig, build_rollout_context`. `lerobot/scripts/lerobot_rollout.py::rollout` shows the sequence: build the context, then strategy setup, run and teardown.
- Subclass `lerobot.rollout.strategies.core.RolloutStrategy`; see `base.py` for the control loop and `episodic.py` for episode recording. The subclass runs one trial per start/stop and records episodes tagged with the trial id.
- Switching arms requires reloading the policy. Measure the cost and design around it.
- Consider proposing an upstream PR that makes rollout strategies pluggable.

**Real-time chunking (RTC) settings** live in the inference config:
- `inference.type=rtc`
- `inference.queue_threshold` (default 30)
- `inference.rtc.execution_horizon` (default 10)
- `inference.rtc.max_guidance_weight`
- `inference.rtc.prefix_attention_schedule`

Don't assume these map one-to-one onto other papers' terms (for example Dream Machines' "reinfer/blend"). Document the mapping only after reading `lerobot/policies/rtc`.

### LeRobot dataset linking (v0.2)

- Datasets use format v3.0 (`lerobot.datasets.dataset_metadata.LeRobotDatasetMetadata`).
- Several episodes share one video file. Link a trial by (repo_id or root, episode_index), and use that episode's timestamps within the video file.
- DAgger rollouts tag correction frames `intervention=True`. When that tag is present, use it to report interventions per trial.

### Reward models (v0.3)

- `lerobot.rewards` exports `PreTrainedRewardModel`, `make_reward_model`, `make_reward_pre_post_processors`, and configs for Robometer, TOPReward and SARM.
- See `lerobot/rewards/*/compute_rabc_weights.py` for how they are used for labeling.
- Auto-labels are suggestions only; the human label is the label. Track agreement with Cohen's κ.

### REST API and client

- **API:** `/api/v1`, documented with OpenAPI.
- **Client:** `fieldtrial.client.Client` lets any runtime (a ROS 2 node, a custom script, C++ over HTTP) get the next trial, report start and stop, submit an outcome and attach media.
- **Live updates:** a Server-Sent Events (SSE) endpoint.

---

## 14. Operator console

The console is mobile-first, server-rendered, and works offline on a LAN.

**Serving and security:**
- `fieldtrial serve` binds to 127.0.0.1 by default.
- `--lan` binds 0.0.0.0, generates a random token, and prints the URL and a terminal QR code. In LAN mode the token is required on every request, and mutations carry a CSRF token.

**Screens:**
- Studies list
- Study overview: design summary, progress, sessions. While blinded, it shows no per-arm results.
- Session start: operator, rig, and the `rig_checklist`
- **Trial console**
- Trial history: edits are allowed and logged as events
- Report

**Trial console:**
- Header such as "Trial 17/80 · block 9 · slot=17", the arm's blind code, and the reset instructions, with optional calibration media.
- A large Start/Stop button and a timer with a timeout warning.
- After Stop:
  - stage buttons: "none" plus one per stage; success follows from reaching the success stage
  - termination reason
  - failure-tag chips and a note
  - an Invalid button that asks for a reason
  - Confirm, which moves to the next trial
- Undo of the last trial for 10 seconds.

**Keyboard:** Space starts and stops, 0–9 set the stage, Enter confirms, I marks invalid, U undoes. USB foot pedals emulate keys, so the bindings must be configurable.

**Live mirror:** a second screen shows the same state over SSE.

**Accessibility:** touch targets of at least 48 px, high contrast, light and dark themes.

**Concurrency:** optimistic concurrency and idempotency keys, so two devices or a double tap can't corrupt a trial.

---

## 15. Analysis and reports

### Results model

`analysis.engine` builds a versioned `Results` model (pydantic, with its JSON schema exported to the docs). It contains:
- the design summary and hash
- per-arm counts and CIs
- the primary test and secondary analyses
- stage funnels and timing
- the per-condition outcome matrix
- session drift and invalid-trial counts
- deviations: amendments, early unblinding, trials run out of order
- provenance: software versions, policy fingerprints, seeds

### Choice of primary analysis

The primary analysis follows from the locked design. Complete blocks get the paired analysis; anything else gets the independent analysis. The independent-sample analysis is always reported as well, as a sensitivity check.

### Wording rules

All report text comes from `analysis.wording`, one tested template per situation.

**Significant result:**

> "q50 succeeded in 92.5% of trials (74/80; 95% CI 84.6–96.5%) vs 75.8% (91/120) for baseline: +16.7 pp (95% CI +6.2 to +26.0; Boschloo p = 0.0020)."

**Not significant:**

> "No significant difference detected: +14.2 pp (95% CI −0.5 to +24.5; p = 0.056). With 40 and 120 trials, this study had 80% power only for differences of at least X pp."

Compute X with `mde`.

**Never write:** "trend toward significance", "almost significant", or "better" without a rejected primary test.

### Charts

matplotlib, rendered as inline SVG, using the Okabe–Ito colorblind-safe palette:
- success rate with CI per arm
- stacked bars of the furthest stage reached
- stage funnel
- cumulative success-time curves
- forest plot of the differences
- condition × arm outcome heatmap
- per-session success over time

### Outputs

- a self-contained `report.html` that makes no external requests
- `report.md` for GitHub and PRs
- `results.json`

---

## 16. CLI (Typer + Rich; `--json` on every command; exit codes 0 ok / 1 error / 2 validation)

**Calculators (Milestone 1):**

```
fieldtrial ci 36/40 [--method wilson|clopper-pearson|jeffreys|agresti-coull] [--level 0.95]
fieldtrial compare 74/80 91/120 [--test boschloo|fisher] [--alternative two-sided|greater|less]
fieldtrial paired --b 10 --c 2 [--n 40]          # exact McNemar; Tango CI when --n is given
fieldtrial power --p1 0.76 --p2 0.90 [--alpha 0.05] [--power 0.8] [--method pooled-z|fleiss-cc|arcsine]
fieldtrial mde --p1 0.76 --n 40
fieldtrial adjust 0.00893 0.00302 0.01624 --method holm|bonferroni|bh
```

**Studies (Milestones 2–4):**

```
fieldtrial init DIR [--template basic|checkpoint-ladder|serving-sweep]
fieldtrial validate DIR
fieldtrial plan DIR                # validation + schedule preview + power/MDE for the planned n
fieldtrial lock DIR
fieldtrial amend DIR --reason "..."
fieldtrial serve DIR [--lan] [--port 8765]
fieldtrial status DIR              # blinded progress
fieldtrial unblind DIR             # logged
fieldtrial simulate DIR --rates baseline=0.76,q50=0.90 [--seed 1]   # sim runner + auto-operator
fieldtrial analyze DIR [--json]
fieldtrial report DIR [--format html|md|json] [--out PATH]
fieldtrial import DIR results.csv [--map arm=policy,success=ok]
fieldtrial export DIR --format csv|jsonl
fieldtrial demo                    # temp study + sim runner + auto-operator, opens console and report
fieldtrial doctor                  # environment and optional-dependency checks
```

**Milestone 1 acceptance:**

| Command | Expected output |
|---|---|
| `fieldtrial ci 36/40` | 0.900, Wilson 95% CI [0.7695, 0.9604] |
| `fieldtrial compare 50/80 13/40 --test fisher` | +30.0 pp, Newcombe [+11.0, +45.8], p = 0.0034 |
| `fieldtrial power --p1 0.76 --p2 0.90` | 112 per arm (pooled-z) |
| `fieldtrial mde --p1 0.76 --n 40` | +21.1 pp / −30.1 pp |

---

## 17. Testing

- **Statistics:** golden tests (§12) and Hypothesis properties:
  - the interval contains the estimate, and low ≤ high
  - symmetry: `ci(k, n)` mirrors `ci(n−k, n)`
  - intervals widen as the level rises
  - Holm-adjusted ≥ raw, and monotone
  - sample size shrinks as |p1 − p2| grows
- **Design:**
  - schedule determinism per seed (snapshots)
  - balance invariants
  - hash stability: cosmetic edits leave the hash unchanged, substantive edits change it
- **Store and services:**
  - every write happens in one transaction with its event
  - optimistic concurrency
  - rescheduling logic
- **Web and API:** a FastAPI TestClient test for every route, including HTMX partials.
- **End-to-end:**
  - the sim runner plus auto-operator fills a study, and `analyze` recovers the true rates within their CIs
  - the report renders with no external URLs, asserted by scanning the HTML
- **Coverage and isolation:** at least 90% overall and 98% for `stats`. No network access in tests.

---

## 18. Docs and examples

**README:**
- the pitch
- a 30-second demo: `uvx fieldtrial compare 74/80 91/120`, then `fieldtrial demo`
- screenshot and GIF placeholders
- quickstart
- how it fits with LeRobot and openpi
- how to cite

**Docs site:**
- **Quickstart.**
- **Concepts:** arms, conditions and blocks, rubric and stages, blinding, pre-registration, invalid trials.
- **Guides:**
  - How many rollouts do I need?
  - Comparing two checkpoints
  - Checkpoint ladders
  - Serving-config sweeps
  - Using fieldtrial with LeRobot
  - Using fieldtrial with openpi
  - Custom runtimes via REST
- **Statistics reference:** one page per method with the formula, when to use it, and references. Include Kress-Gazit et al. 2024, *Robot Learning as an Empirical Science: Best Practices for Policy Evaluation*, arXiv:2409.09491.
- **API reference.**

**Example: `examples/dream-machines-pi05/`**
- `published_counts.csv` (Appendix A).
- `reanalysis.py`, which writes a Markdown report covering:
  - which of the post's comparisons are statistically resolved, with a CI for each
  - the Holm-adjusted serving-settings sweep
  - the number of trials needed to resolve the open questions
- Attribute and link the source.
- State that these were separately-run evaluations, not one randomized study, so session-to-session drift is uncontrolled.

---

## 19. Tooling, CI, release

**pre-commit:** ruff (lint and format), mypy, end-of-file and whitespace fixers, check-yaml, and a uv lock check.

**GitHub Actions:**
- `ci.yml`:
  - matrix: Python 3.11 / 3.12 / 3.13 × Ubuntu / macOS
  - steps: `uv sync`, ruff, mypy, `lint-imports`, and `pytest -m "not slow"` with coverage
  - a nightly or manual job runs the `slow` statistical tests
- `docs.yml`: build docs with `--strict` and deploy to Pages on `main`.
- `release.yml`: on a `v*` tag, build the sdist and wheel, publish to PyPI via trusted publishing, and create a GitHub release with the changelog section.

**Versioning:** SemVer (0.x may break things) and a Keep a Changelog-style CHANGELOG.

**Templates:**
- issue templates: bug, feature request, statistics question
- PR template with a checklist: tests, docs, changelog

---

## 20. Milestones and definition of done

### M0: Bootstrap

**Build:**
- uv project with src layout
- `pyproject.toml`: metadata, extras, and the entry point `fieldtrial = "fieldtrial.cli.main:app"`
- ruff, mypy and pytest configuration; pre-commit; CI workflows
- LICENSE, README skeleton, CONTRIBUTING, CODE_OF_CONDUCT (Contributor Covenant), SECURITY, CHANGELOG
- mkdocs skeleton, `docs/PLAN.md`, `docs/ROADMAP.md`, `CLAUDE.md`
- `fieldtrial --version`

**Done when:**
- `uv run pytest`, `uv run ruff check .`, `uv run mypy src` and `uv run mkdocs build --strict` all pass
- the first commit is made

### M1: Statistics core and calculator CLI (release 0.1.0a1)

**Build:**
- the §12 modules: proportions, compare, paired, multiplicity, power/MDE (including exact power), bayes
- the calculator commands from §16
- a docs page per method

Ordinal, timing, stratified and drift analyses come in M2 with the analysis engine.

**Done when:**
- golden and property tests pass, and the slow tests pass locally
- `stats` coverage is at least 98%
- the acceptance commands print the expected numbers
- `uv build` produces a wheel

I publish 0.1.0a1. The `0.1.0.dev0` placeholder has already reserved the name.

### M2: Study design, storage and analysis (no UI yet)

**Build:**
- the `study.yaml` schema, loader and helpful errors, plus three templates
- the randomized complete block schedule with balancing
- blinding codes
- hash, lock and amend
- SQLite store with an Alembic baseline migration
- services and the event log
- CSV import and export
- the analysis engine: primary, secondary and sensitivity analyses, ordinal, timing, drift flags
- the wording module and a v0 Markdown report
- CLI: init, validate, plan, lock, amend, import, export, analyze, status, unblind, simulate

**Done when:**
- a study can be designed, locked, filled by `simulate` or CSV import, and analyzed end to end
- a fixture study's Markdown report reproduces the golden numbers
- schedule snapshot tests and the end-to-end simulation test pass

### M3: Operator console and API

**Build:**
- FastAPI app with HTMX templates and vendored assets
- session and trial flows
- keyboard and pedal bindings
- SSE live updates
- LAN token and QR code
- idempotency and optimistic concurrency
- REST API v1 and `fieldtrial.client`
- the `serve` and `demo` commands
- manual and sim runners wired into the console

**Done when:**
- `fieldtrial demo` works end to end on a laptop
- there is a TestClient test for every route
- a test confirms pages make no external requests
- you give me a short manual checklist for testing the console on a phone

### M4: Reports, docs and example (release v0.1.0)

**Build:**
- the HTML report with the §15 charts and the `report` command
- the Dream Machines re-analysis example
- complete docs: quickstart, concepts, guides, statistics reference, API
- a polished README and the CHANGELOG
- the release workflow, tested on TestPyPI

**Done when:**
- reports for the demo study and the example are self-contained (asserted)
- the docs build passes in strict mode
- in a clean environment, installing 0.1.0rc1 from TestPyPI with `uvx` and running `fieldtrial demo` works
- I sign off

### M5 and later

Follow §5 for v0.2 and v0.3. Write a fresh plan before starting.

---

## 21. `CLAUDE.md` (create exactly this, then keep it current)

```markdown
# fieldtrial: notes for Claude Code

Statistically rigorous real-world evaluation for robot policies.
Source of truth: docs/PLAN.md (read it when unsure). Roadmap: docs/ROADMAP.md.

## Commands
- Setup: `uv sync --all-extras`
- Fast tests: `uv run pytest -m "not slow"`
- All tests: `uv run pytest`
- Lint and format: `uv run ruff check --fix . && uv run ruff format .`
- Types: `uv run mypy src`
- Import contracts: `uv run lint-imports`
- Docs: `uv run mkdocs serve`
- Try it: `uv run fieldtrial demo`

## Architecture rules
- `fieldtrial.stats` is pure: numpy and scipy only, no I/O, no imports from other fieldtrial modules.
- CLI, web and API call `fieldtrial.services`; nothing else touches the DB.
- Every write is one transaction plus one row in the `event` table.
- Never import torch, lerobot or openpi at module import time. Lazy-import inside runners, behind extras.
- All randomness comes from the study seed via numpy.random.Generator(PCG64(seed)).

## Statistics rules
- Any new or changed stats function needs:
  - a golden test against an independent reference
  - a property test
  - a docs entry with the formula and a reference
- The primary analysis comes from the locked design. Never choose tests after seeing the data.
- Report text goes through `fieldtrial.analysis.wording` only. Never call an arm "better" without a rejected pre-registered test.

## Conventions
- Python ≥3.11, full type hints.
- pydantic v2 for schemas; frozen dataclasses for stats results.
- Conventional commits. Update CHANGELOG.md under "Unreleased" for user-visible changes.
- No network in tests, no telemetry, web assets vendored (no CDN).
- Verify third-party APIs (LeRobot, openpi) against the installed source before using them.
- Out-of-scope ideas go to docs/ROADMAP.md, not into the code.
```

---

## Appendix A: Dream Machines' published counts

Source: https://dream-machines.eu/blog/pi05-fine-tuning (September 2026).
- Each row is end-to-end success out of trials.
- "R"/"B" are Dream Machines' own terms for their serving settings: reinfer and blend.
- The 4h five-scene dataset is the same one used in the data-quantity comparison.

```csv
experiment,arm,successes,trials
lora_vs_full,full_finetune,91,120
lora_vs_full,lora_r64_ebs256,18,40
lora_vs_full,lora_r64_ebs64,13,40
lora_vs_full,lora_r256_ebs64,21,40
learning_rate,lr_2e-5,57,80
learning_rate,lr_5e-5_baseline,91,120
learning_rate,lr_1e-4,51,80
batch_size,bs_64,54,80
batch_size,bs_256_baseline,91,120
batch_size,bs_2048,50,80
action_space_and_augmentation,baseline,91,120
action_space_and_augmentation,relative_joint_targets,29,40
action_space_and_augmentation,image_augmentation,30,40
operators_10h,operator1_vr,49,80
operators_10h,operator2_leader_arms,12,40
operators_10h,mixed,51,80
data_quantity,4h,50,80
data_quantity,10h,51,80
data_quantity,21h,91,120
scene_diversity_4h,lightbox_only,13,40
scene_diversity_4h,five_scenes,50,80
data_quality,21h_baseline,91,120
data_quality,21h_plus_1h_hq,36,40
data_quality,1h_hq_only,11,40
interventions_standard_positions,1h_hq_only,11,40
interventions_standard_positions,1h_hq_plus_interventions,35,40
interventions_diverse_positions,1h_hq_only,21,200
interventions_diverse_positions,1h_hq_plus_interventions,19,40
serving_sweep_21h,baseline_R30_B16,91,120
serving_sweep_21h,sync_no_rtc,21,40
serving_sweep_21h,R25_B50,20,40
serving_sweep_21h,R50_B50,22,40
serving_sweep_21h,R50_B1,32,40
serving_sweep_21h,R50_B16,35,40
serving_sweep_21h,R50_B20,74,80
serving_sweep_21h,R50_B25,68,80
serving_sweep_21h,R50_B25_compiled,29,40
serving_best_on_21h_plus_1h,R50_B20,39,40
initialization,pi05_pretrained,91,120
initialization,paligemma_only,0,40
```

---

## Appendix B: Decision log

| Date | Decision |
|---|---|
| 2026-10-01 | Name `fieldtrial`; Apache-2.0; Python ≥3.11; uv + hatchling; SQLAlchemy 2 + Alembic on SQLite; FastAPI + Jinja2 + HTMX (no Node); matplotlib SVG charts. |
| 2026-10-01 | Primary tests: Boschloo (independent arms), exact McNemar (paired); Wilson intervals; Holm for multiple comparisons; Bayesian results are descriptive only. |
| 2026-10-01 | Manual and sim runners in v0.1. Command runner, openpi router and LeRobot dataset linking in v0.2. In-process LeRobot runner and reward-model pre-labeling in v0.3. |
| 2026-10-01 | LeRobot rollout strategies can't be registered externally (hard-coded dispatch in 0.6.1), so the LeRobot runner will use `build_rollout_context` and its own `RolloutStrategy` subclass. |
| 2026-10-01 | The repository is `rokbenko/fieldtrial`, with `main` as the default branch. Claude Code works on a `claude/` branch and pushes it, because its cloud sessions are temporary. PRs, tags and other branches need the maintainer's OK. This replaces "Do not push" in §0. |
| 2026-10-01 | Author and copyright holder: Rok Benko. Project metadata carries no email. Security reports go through GitHub's private vulnerability reporting. |
| 2026-10-01 | The PyPI name is reserved now with a placeholder, `0.1.0.dev0`, that contains no code. As a pre-release, pip and uv never pick it over 0.1.0a1 or later. The maintainer uploads it, because the Claude Code container's network policy blocks upload.pypi.org. |
| 2026-10-01 | Integration extras (`openpi`, `capture`, `lerobot`, `rewards`) are declared in the milestone whose code uses them, and core dependencies are added in the milestone that first imports them. Declaring `lerobot` early would pull torch into `uv sync --all-extras` and pin the dev lockfile to lerobot's `numpy<2.3`. |
| 2026-10-01 | Exact power for Boschloo and McNemar comes from enumerating the outcome space rather than from simulation; seeded simulation remains as a cross-check. |
| 2026-10-01 | mypy runs in strict mode for the whole package, because mypy can't enable `strict` per module. |
| 2026-10-01 | The PyPI name is reserved: `0.1.0.dev0` was published on 2026-10-01. |
| 2026-10-01 | Corrected the §12 Boschloo golden values and the §1 and §15 examples. The original values came from a transposed table (arms as rows), but `scipy.stats.boschloo_exact` treats each column as one arm. With arms as columns: 0.00228 (was 0.00201), 0.05574 (was 0.05278) and 0.00196 (was 0.00179). No conclusion changes. Fisher's test is unaffected because it is symmetric under transposition. |

---

**First action now:** do steps 1–3 of §0. Create `docs/PLAN.md` and `CLAUDE.md`, then show me your one-page plan for M0 and M1, with any blocking questions.
