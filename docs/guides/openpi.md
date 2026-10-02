# Using fieldtrial with openpi

openpi serves policies over a websocket: the robot client sends observations and receives
actions. fieldtrial fits around that loop.

## Real blinding: the openpi router

Point the robot's client at fieldtrial's router once, and it sends each trial's requests
to that trial's policy server. Nobody at the robot needs to know which arm runs. See
[Real blinding with runners](runners.md#the-openpi-router) for the setup.

## Without the router: one server per arm

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
