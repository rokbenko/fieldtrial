# Re-analysis of Dream Machines' pi0.5 fine-tuning results

Dream Machines published a detailed account of fine-tuning pi0.5 for a bimanual
pick-handover-insert task, with the success counts of every arm they evaluated:
[*Fine-tuning pi0.5*](https://dream-machines.eu/blog/pi05-fine-tuning) (September 2026).
All counts in `published_counts.csv` are theirs; this folder only re-analyzes them.

**[Read the re-analysis: REPORT.md](REPORT.md)**

## What it shows

- Which of the post's comparisons are statistically resolved, each with a 95% interval
  for the difference (Boschloo's exact test, Newcombe interval, Holm's adjustment within
  each experiment).
- The serving-settings sweep after Holm's adjustment: three of eight settings differ from
  the default, one of them (`R50_B20`) in the positive direction.
- For each open question, the number of trials per arm that would resolve it if the
  observed rates were the true ones.

## Caveats

- These were **separately run evaluations, not one randomized study**. Arms were
  evaluated at different times, so session-to-session drift (lighting, gripper wear,
  operator) is uncontrolled and can look like an effect. A fieldtrial study runs the arms
  interleaved in randomized blocks for exactly this reason.
- The 91/120 baseline is one set of runs reused across several experiments, so those
  comparisons are not independent of each other.
- "Not resolved" means the data cannot tell the arms apart, not that they are equal.
- The counts were transcribed from the post into `docs/PLAN.md` (Appendix A) and from
  there into `published_counts.csv`. If the post is updated, update the CSV and rerun.

## Reproduce

```console
uv run python examples/dream-machines-pi05/reanalysis.py
```

The script uses only `fieldtrial.stats` and `fieldtrial.analysis.wording`, runs in a few
seconds, and rewrites `REPORT.md`. A test checks that the committed report matches the
script's output.
