# Using fieldtrial with openpi

openpi serves policies over a websocket: the robot client sends observations and receives
actions. fieldtrial fits around that loop.

## Today: one server per arm

1. **Describe each arm** in `study.yaml` with what is needed to start its policy server:
   the checkpoint and any serving options, in the free-form `policy` and `serving`
   dictionaries.
2. **Start one policy server per arm**, each on its own port, labelled by blind code
   only.
3. **Run the console** (`fieldtrial serve my-study --lan`). For each trial, point the
   robot client at the server for the blind code shown, press Start when the robot
   starts and Stop when it ends, and label the outcome.

Pointing the client at a different server is a visible step, so this is only
half-blind.

## Custom clients

If your robot client is a script you control, it can run the study itself through the
[REST API](../api.md): ask for the next trial, switch to that arm's server, run, and report
the outcome. See [Custom runtimes](custom-runtimes.md).

## Coming next (v0.2)

An **openpi router**: a websocket proxy between the robot client and one upstream server
per arm. The client connects to the router and never changes; the router forwards each
frame to the arm of the current trial without decoding it, records per-request latency
per arm, and refuses to start if the arms' metadata differ. That gives real blinding:
nobody at the robot knows which arm is running.
