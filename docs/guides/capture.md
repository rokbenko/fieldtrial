# Evaluation camera and rig checks

An evaluation camera records every trial, so an outcome can be checked later. A rig
reference photo catches a rig that drifted between sessions: a bumped camera, different
light, a moved fixture.

```yaml
capture:
  camera: 0                # OpenCV device index, or a stream URL (rtsp://..., http://...)
  fps: 15
  width: 1280              # optional; the camera's default otherwise
  height: 720
  record: true             # record every trial from Start to Stop
  drift:
    max_shift_px: 8        # flag a rig check above these
    max_brightness_change: 0.15
    min_similarity: 0.75
    every_trials: 20       # also check the rig from the camera every 20 trials
```

The `capture` section describes the setup, not the design, so it is left out of the
design hash: moving the camera to another port does not count as a change to the study.

## Recording trials

Recording needs the `capture` extra (`pip install 'fieldtrial[capture]'`: OpenCV reads the
camera, PyAV encodes the clips). When `fieldtrial serve` runs a trial, the camera records
from **Start** to **Stop**. The clip is attached to the trial as `media/<trial>/camera.mp4`
(H.264). The console shows "camera recording" while it runs.

A camera that fails never blocks a trial. The console shows the error on its runner line,
and the trial runs and is labelled as usual.

## Rig checks

Take a reference photo of the rig as it should look, from a fixed position (ideally the
evaluation camera itself), and set it once:

```console
$ fieldtrial rig-check my-study reference.jpg --set-reference
$ fieldtrial rig-check my-study --camera --set-reference     # or straight from the camera
```

Then check the rig:

- when starting a session in the console, by attaching a phone photo (the form offers it
  once a reference exists), or automatically from the camera
- every `drift.every_trials` trials from the camera
- any time from the command line: `fieldtrial rig-check my-study photo.jpg` or `--camera`

Each check compares the photo with the reference, both reduced to grayscale at about 256
pixels wide:

| Measure | How | Flagged when |
|---|---|---|
| Shift | Phase correlation (Kuglin and Hines, 1975), in reference pixels | above `max_shift_px` |
| Brightness | Relative change of mean brightness | beyond ± `max_brightness_change` |
| Similarity | Structural similarity (SSIM, Wang et al., 2004) after undoing the shift, 7 × 7 windows | below `min_similarity` |

A flagged check shows as a warning in the console. Every report lists it as a deviation
and has a table of all checks. The photos are kept under `rig/checks/`. A flag is a
reason to look at the rig, not proof that it changed. A person walking through the frame
also lowers the similarity.

## References

- Kuglin, C. D. and Hines, D. C. (1975). The phase correlation image alignment method.
  *Proc. IEEE Conference on Cybernetics and Society*, 163–165.
- Wang, Z., Bovik, A. C., Sheikh, H. R. and Simoncelli, E. P. (2004). Image quality
  assessment: from error visibility to structural similarity. *IEEE Transactions on Image
  Processing* 13, 600–612.
