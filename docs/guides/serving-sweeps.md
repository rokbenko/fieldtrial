# Serving-config sweeps

The same checkpoint can succeed more or less often depending on how it is served: action
chunking, real-time chunking (RTC) settings, compilation. A sweep compares several
serving configurations of one policy.

```console
$ fieldtrial init sweep --template serving-sweep
```

Serving settings go in each arm's `serving` dictionary, which is passed to the runner
unchanged. With the manual runner, the console shows the operator which blind code to
load; the operator starts the policy with that arm's settings.

The template compares three settings on three objects × ten positions:

```yaml
arms:
  - id: default
    serving: {inference.type: rtc, inference.queue_threshold: 30}
  - id: q50
    serving: {inference.type: rtc, inference.queue_threshold: 50}
  - id: sync
    serving: {inference.type: sync}
```

The analysis is the same as for a [checkpoint ladder](checkpoint-ladders.md): Cochran's Q,
then each setting against the control with Holm's adjustment.

## Lessons from a published sweep

The [Dream Machines re-analysis](https://github.com/rokbenko/fieldtrial/tree/main/examples/dream-machines-pi05)
in the repository applies this to a published sweep of eight serving settings: after
Holm's adjustment only three differ from the default, and several apparent improvements
of 9–12 points would need 170–300 rollouts per arm to resolve. Sweeps find real effects
when the effects are large; small tuning gains need large, interleaved studies.

## Practical tips

- Put the settings you expect to matter most first and keep the sweep small.
- Interleave settings in randomized blocks (the default) rather than running one
  setting per day: drift between days can look exactly like a serving effect.
- When several settings look promising, follow up with a two-arm study between the best
  one and the default, with enough rollouts to resolve the difference.
