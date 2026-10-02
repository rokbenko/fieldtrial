# Command line

Every command accepts `--json` for machine-readable output. Exit codes are 0 on success, 1
when the request cannot be carried out (for example analyzing a blinded study), and 2 on
invalid input.

## Studies

See [Running a study](studies.md) for the full workflow.

| Command | What it does |
|---|---|
| `fieldtrial init DIR [--template basic\|checkpoint-ladder\|crossover-rounds\|serving-sweep]` | Create a study folder with a `study.yaml` |
| `fieldtrial validate DIR` | Check `study.yaml`; every problem with its line |
| `fieldtrial plan DIR [--baseline 0.76]` | Schedule preview and minimum detectable effect |
| `fieldtrial lock DIR` | Freeze the design and randomize the schedule |
| `fieldtrial amend DIR --reason "..."` | Apply `study.yaml` edits as a logged amendment |
| `fieldtrial simulate DIR --rates baseline=0.76,q50=0.90 [--seed 1] [--no-interim]` | Fill the study with simulated trials (running planned interim looks as they come due) |
| `fieldtrial import DIR results.csv [--map arm=policy,success=ok]` | Import trials from a CSV (all or nothing) |
| `fieldtrial status DIR` | Progress; no per-arm results while blinded |
| `fieldtrial unblind DIR [--yes]` | Reveal the blind codes (logged) |
| `fieldtrial check-runners DIR [--api-key KEY]` | Check the study's runner: commands found, policy servers answering with identical metadata (blind codes only) |
| `fieldtrial rig-check DIR [PHOTO \| --camera] [--set-reference]` | Compare a rig photo with the reference photo, or set the reference ([guide](guides/capture.md)) |
| `fieldtrial link-episodes DIR DATASET [--first-episode N] [--map links.csv] [--yes]` | Link trials to LeRobot v3.0 dataset episodes ([guide](guides/lerobot.md#linking-trials-to-dataset-episodes)) |
| `fieldtrial interim DIR` | Run the planned interim look that is due; prints only "continue" or "stop" |
| `fieldtrial analyze DIR` | Run the pre-registered analysis |
| `fieldtrial report DIR [--format md\|json] [--out PATH]` | Write `reports/report.md` or `results.json` |
| `fieldtrial export DIR [--format csv\|jsonl] [--out PATH]` | Export trials (blind codes while blinded) |
| `fieldtrial serve DIR [--lan] [--port 8765]` | Run the [operator console](console.md) and [REST API](api.md) |
| `fieldtrial demo [--dir PATH] [--no-browser]` | A half-run simulated study in the console |

## Calculators

The calculator commands work on plain counts; you don't need a study.

```console
$ uvx fieldtrial ci 36/40
36/40 = 0.900
Wilson 95% CI [0.7695, 0.9604]

$ uvx fieldtrial compare 74/80 91/120
Arm 1: 74/80 = 92.5% (95% CI 84.6%–96.5%)
Arm 2: 91/120 = 75.8% (95% CI 67.4%–82.6%)
Difference +16.7 pp, Newcombe 95% CI [+6.2 pp, +26.0 pp]
Boschloo p = 0.0020 (two-sided); rejects H0 at α = 0.05
Fisher p = 0.0022

$ uvx fieldtrial paired --b 10 --c 2 --n 40
Discordant pairs: b = 10, c = 2
Difference +20.0 pp, Tango 95% CI [+3.5 pp, +36.6 pp]
Exact McNemar p = 0.0386 (two-sided)

$ uvx fieldtrial power --p1 0.76 --p2 0.90
112 per arm (pooled-z)

$ uvx fieldtrial mde --p1 0.76 --n 40
+21.1 pp (to 0.9707) / −30.1 pp (to 0.4588)

$ uvx fieldtrial adjust 0.00893 0.00302 0.01624 --method holm
```

| Command | Options |
|---|---|
| `ci K/N` | `--method wilson\|clopper-pearson\|jeffreys\|agresti-coull`, `--level` |
| `compare K1/N1 K2/N2` | `--test boschloo\|fisher`, `--alternative two-sided\|greater\|less`, `--level`, `--alpha` |
| `paired --b B --c C` | `--n` (adds the Tango CI), `--alternative`, `--level` |
| `power --p1 P1 --p2 P2` | `--alpha`, `--power`, `--method pooled-z\|fleiss-cc\|arcsine`, `--ratio`, `--alternative` |
| `mde --p1 P --n N` | `--n2`, `--alpha`, `--power`, `--method` |
| `adjust P...` | `--method holm\|bonferroni\|bh`, `--alpha` |

`--alternative greater` always means *arm 1 is higher*.
