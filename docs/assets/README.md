# How the images in README.md were made

Every image here comes from fieldtrial itself: the simulated robot, published counts, or
`fieldtrial.stats`. None of them was recorded at a real robot.

| File | What it is | How it was made |
|---|---|---|
| `fieldtrial.gif` | The console on a phone and a second screen, then the report | `fieldtrial demo` on 2026-10-02, recorded with Playwright (below) |
| `console-running.png`, `console-label.png` | The console on a phone during and after a trial | `fieldtrial demo`, Playwright screenshots at 390 × 780 |
| `report.png` | The top of an HTML report | `fieldtrial demo` after unblinding, Playwright screenshot |
| `rollouts.png` | Interval width by rollouts, rollouts needed per improvement | `make_readme_charts.py` |
| `dream-machines.png` | Every comparison of the Dream Machines re-analysis | `make_readme_charts.py`, from `examples/dream-machines-pi05/published_counts.csv` |
| `anytime.png` | The report's anytime-valid section | A simulated study (below) |
| `best-arm.png` | The report's best-arm selection section | A simulated study (below) |

## The recording

`fieldtrial demo` creates a blinded two-arm study whose simulated arms truly succeed 76%
and 90% of the time and runs half of it. `record_hero.mjs` opens two browser contexts, a
phone (390 × 780) and a desk screen (960 × 780), and records both as video. The phone
starts a session and runs five trials from the keyboard, labelling three as successes and
two as partial with a failure tag; the desk screen shows the console's live mirror. The
remaining trials are then run quickly (cut from the GIF), the desk screen confirms the
logged unblinding and scrolls through the report.

```bash
fieldtrial demo --dir /tmp/demo-study --port 8798 --no-browser &
node docs/assets/record_hero.mjs /tmp/hero        # writes phone/*.webm, desk/*.webm, marks.json
```

`marks.json` holds the seconds at which each part starts. The GIF is the two videos side
by side, the trials at twice their speed and the report at 1.4 times, at 10 frames per
second and 1000 pixels wide, with a 128-colour palette:

```bash
ffmpeg -i phone.webm -i desk.webm -filter_complex "
[0:v]trim=0.4:28.6,setpts=(PTS-STARTPTS)/2,fps=10,pad=410:850:10:60:color=0x111827[pa];
[1:v]trim=0.4:28.6,setpts=(PTS-STARTPTS)/2,fps=10,pad=980:850:10:60:color=0x111827[da];
[pa][da]hstack[a];
[0:v]trim=33.4:33.5,setpts=PTS-STARTPTS,loop=200:1:0,fps=10,trim=0:8,pad=410:850:10:60:color=0x111827[pb];
[1:v]trim=33.5:44.5,setpts=(PTS-STARTPTS)/1.4,fps=10,pad=980:850:10:60:color=0x111827[db];
[pb][db]hstack[b];
[a][b]concat=n=2:v=1,scale=1000:-1:flags=lanczos[out]" -map "[out]" hero.mp4
ffmpeg -i hero.mp4 -vf "fps=10,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" fieldtrial.gif
```

The captions above each half ("At the robot", "Second screen", "At the desk") were added
with ffmpeg's `drawtext` filter in the same command.

## The simulated studies

```bash
fieldtrial init anytime                  # then set range: [1, 60] and stopping: {rule: anytime}
fieldtrial lock anytime
fieldtrial simulate anytime --rates baseline=0.55,q50=0.88 --seed 7
fieldtrial unblind anytime --yes && fieldtrial report anytime

fieldtrial init sweep --template best-arm
fieldtrial lock sweep
fieldtrial simulate sweep --rates q30=0.62,q50=0.82,q70=0.88,sync=0.45 --seed 3
fieldtrial unblind sweep --yes && fieldtrial report sweep
```

The two report sections were captured with Playwright at twice the pixel density and
scaled to 1100 pixels wide. Each is a single simulation, picked to show what the charts
look like; the simulated power of both designs is in `docs/stats/step-evaluation.md` and
`docs/stats/selection.md`.

## The charts

```bash
uv run python docs/assets/make_readme_charts.py
```
