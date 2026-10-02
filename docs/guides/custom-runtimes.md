# Custom runtimes via REST

Any program that can make HTTP requests can run a study: a ROS 2 node, a rollout script,
a C++ controller. The loop is:

1. Start a session.
2. Ask for the next trial (`up_next`): its blind code and condition.
3. Load that arm, set up the condition, start the trial.
4. Stop the clock when the robot stops, then report the furthest stage reached and why
   it ended (or mark the trial invalid).
5. Repeat until `up_next` is empty, then end the session.

In Python, `fieldtrial.client` does this with the standard library only:

```python
from fieldtrial.client import Client

api = Client("http://192.168.1.20:8765", study="my-study", token="...")  # token from --lan
state = api.start_session(operator="ros-node", rig="rig-1")
session = state.session.session_id

while (slot := api.session(session).up_next) is not None:
    load_arm(slot.blind_code)  # your code: switch policy by blind code
    reset_scene(slot.factors)  # your code: e.g. {"slot": 17}
    trial = api.start_trial(slot.slot_id, session)
    outcome = run_policy()  # your code: returns stage reached, reason
    stopped = api.stop_trial(trial.trial_id, expected_version=trial.version)
    if outcome.fault:
        api.invalidate_trial(trial.trial_id, outcome.fault, expected_version=stopped.version)
    else:
        api.complete_trial(
            trial.trial_id,
            stage=outcome.stage,  # a stage id from the rubric, or None
            termination=outcome.reason,  # success, timeout, stuck, ...
            expected_version=stopped.version,
        )

api.end_session(session)
```

While the study is blinded the API never reveals which arm a blind code is, so your
runtime needs its own mapping from blind code to policy, prepared by someone who is not
operating the robot.

Other languages can use the same HTTP API directly; see [REST API and client](../api.md)
for the rules (idempotency keys, versions, the client header) and the
[OpenAPI document](../reference/openapi.json) for every route. A human can follow the
same session in the console's mirror screen while your runtime runs it.
