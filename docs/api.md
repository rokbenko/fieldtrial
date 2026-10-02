# REST API and Python client

Custom runtimes (a ROS 2 node, a rollout script, C++ over HTTP) can run a study through the
REST API that `fieldtrial serve` provides at `/api/v1`. The
[OpenAPI document](reference/openapi.json) describes every route; the server also serves it
at `/api/v1/openapi.json`.

## Python client

`fieldtrial.client` uses only the standard library, so it adds no dependencies to your
runtime:

```python
from fieldtrial.client import Client

api = Client("http://127.0.0.1:8765", study="my-study")
state = api.start_session(operator="ana", rig="rig-1")
session = state.session.session_id

while (slot := api.session(session).up_next) is not None:
    trial = api.start_trial(slot.slot_id, session)
    # ... load arm slot.blind_code, run the policy, judge the outcome ...
    stopped = api.stop_trial(trial.trial_id, expected_version=trial.version)
    api.complete_trial(
        trial.trial_id,
        stage="clean",  # furthest stage reached, or None
        termination="success",
        expected_version=stopped.version,
    )

api.end_session(session)
```

Other calls: `studies()`, `status()`, `next()`, `trial()`, `trials()`,
`invalidate_trial(trial_id, reason)`, `undo(trial_id)`, `attach_media(trial_id, path)`, and
for group-sequential studies `interim_status()` and `run_interim()`
(`GET`/`POST /api/v1/studies/{study}/interim`). An interim look returns only "continue" or
"stop", and a stop cancels the remaining trials. `run_interim()` is never retried
automatically, because a look is not idempotent. For anytime-valid and best-arm
selection studies, `adaptive_status()` (`GET /api/v1/studies/{study}/adaptive`) returns
the rule, the complete blocks, whether the study stopped and the blind codes of dropped
arms; their looks run by themselves when `complete_trial()` finishes a block.
Errors raise `ApiError` with the HTTP `status` and a `detail` message.

If the study's arms use a [switching runner](guides/runners.md), starting, stopping,
completing and voiding trials through the API drives it exactly as the console does.

## Rules for any client

- **Writes** need an `X-Fieldtrial-Client` header (any value); browsers cannot send it
  from other sites. In LAN mode, also send `Authorization: Bearer <token>`.
- **Idempotency:** send an `Idempotency-Key` header (up to 64 characters, unique per
  intended write) and retry freely; a repeat returns the first result.
- **Concurrency:** send the trial's `version` back as `expected_version`. If the trial
  changed in the meantime, you get 409 and nothing is written.
- **Blinding:** while a study is blinded, `arm`, `policy` and `serving` are `null`; only
  `blind_code` identifies the arm.
- **Errors** are JSON `{"detail": "..."}`: 400 when the study cannot accept the request,
  404 for an unknown study, 409 for a version conflict, 422 for a malformed body.

## Live updates

`GET /api/v1/studies/{study}/events` is a Server-Sent Events stream with one `changed` event
per change to the study (its `data` holds the event kind and time). Add `?once=true` to get
what is new and close, which suits simple polling with `after=<last event id>`.
